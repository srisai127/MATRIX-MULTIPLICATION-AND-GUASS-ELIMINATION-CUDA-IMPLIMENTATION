
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

#define EPSILON 1e-6

#define CUDA_CHECK(call)                                                   \
{                                                                          \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess)                                              \
    {                                                                      \
        cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;       \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

double getTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
    Generate a Strictly Diagonally Dominant matrix.

    Off-diagonal values are integers in [-10,10].
    Diagonal value is selected so that:

        |A[i][i]| > sum of |A[i][j]|, j != i

    B is initialized to 100.
*/
void generateSystem(vector<double>& A, vector<double>& B, int n)
{
    srand(12345);

    A.assign(n * n, 0.0);
    B.assign(n, 100.0);

    for (int i = 0; i < n; i++)
    {
        int rowSum = 0;

        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                int value = (rand() % 21) - 10;
                A[i * n + j] = value;
                rowSum += abs(value);
            }
        }

        /*
            Make the matrix strictly diagonally dominant.
            Diagonal is larger than the absolute sum of
            all other elements in that row.
        */
        A[i * n + i] = rowSum + 1;
    }
}

/*
    Sequential Gaussian Elimination.

    Forward elimination followed by back substitution.
*/
void gaussianSequential(
    const vector<double>& A_input,
    const vector<double>& B_input,
    vector<double>& X,
    int n)
{
    vector<double> A = A_input;
    vector<double> B = B_input;

    // Forward elimination
    for (int k = 0; k < n - 1; k++)
    {
        double pivot = A[k * n + k];

        for (int i = k + 1; i < n; i++)
        {
            double factor = A[i * n + k] / pivot;

            A[i * n + k] = 0.0;

            for (int j = k + 1; j < n; j++)
            {
                A[i * n + j] -= factor * A[k * n + j];
            }

            B[i] -= factor * B[k];
        }
    }

    // Back substitution
    X.assign(n, 0.0);

    for (int i = n - 1; i >= 0; i--)
    {
        double sum = B[i];

        for (int j = i + 1; j < n; j++)
        {
            sum -= A[i * n + j] * X[j];
        }

        X[i] = sum / A[i * n + i];
    }
}


/*
    CUDA kernel.

    Each CUDA thread calculates one element of the
    current row update.

    For pivot k:

        A[i][j] = A[i][j] - factor * A[k][j]

    The rows below the pivot are processed in parallel.
*/
__global__ void gaussianEliminationKernel(
    double* A,
    double* B,
    int n,
    int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    int col = blockIdx.x * blockDim.x + threadIdx.x + k + 1;

    if (row < n && col < n)
    {
        double pivot = A[k * n + k];

        double factor = A[row * n + k] / pivot;

        A[row * n + col] -= factor * A[k * n + col];
    }
}

/*
    Separate kernel to eliminate the first column of
    each row below the pivot and update B.

    This is done once per row.
*/
__global__ void calculateFactorsKernel(
    double* A,
    double* B,
    int n,
    int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x + k + 1;

    if (row < n)
    {
        double pivot = A[k * n + k];
        double factor = A[row * n + k] / pivot;

        A[row * n + k] = 0.0;
        B[row] -= factor * B[k];
    }
}

/*
    CUDA Gaussian elimination.

    The pivot loop itself remains sequential.

    For every pivot:
      1. Calculate row factors in parallel.
      2. Update matrix elements in parallel.
*/
void gaussianCUDA(
    const vector<double>& A_input,
    const vector<double>& B_input,
    vector<double>& X,
    int n,
    int threads)
{
    double* d_A;
    double* d_B;

    size_t matrixSize = (size_t)n * n * sizeof(double);
    size_t vectorSize = (size_t)n * sizeof(double);

    CUDA_CHECK(cudaMalloc((void**)&d_A, matrixSize));
    CUDA_CHECK(cudaMalloc((void**)&d_B, vectorSize));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        A_input.data(),
        matrixSize,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        B_input.data(),
        vectorSize,
        cudaMemcpyHostToDevice));

    /*
        We use a 2D grid for matrix updates.

        threads x threads threads per block.
    */
    dim3 block(threads, threads);

    for (int k = 0; k < n - 1; k++)
    {
        // Calculate elimination factors and update B.
        int rowsRemaining = n - k - 1;

        int blocks1D =
            (rowsRemaining + threads - 1) / threads;

        calculateFactorsKernel<<<blocks1D, threads>>>(
            d_A,
            d_B,
            n,
            k);

        CUDA_CHECK(cudaGetLastError());

        /*
            Update the remaining matrix.

            Rows = n-k-1
            Columns = n-k-1
        */
        dim3 grid(
            (n - k - 1 + threads - 1) / threads,
            (n - k - 1 + threads - 1) / threads);

        gaussianEliminationKernel<<<grid, block>>>(
            d_A,
            d_B,
            n,
            k);

        CUDA_CHECK(cudaGetLastError());

        /*
            Synchronize after each pivot because the next
            pivot depends on the completed current step.
        */
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /*
        Copy the upper triangular matrix and B back to CPU.
    */
    vector<double> A(n * n);
    vector<double> B(n);

    CUDA_CHECK(cudaMemcpy(
        A.data(),
        d_A,
        matrixSize,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        B.data(),
        d_B,
        vectorSize,
        cudaMemcpyDeviceToHost));

    /*
        Back substitution on CPU.
    */
    X.assign(n, 0.0);

    for (int i = n - 1; i >= 0; i--)
    {
        double sum = B[i];

        for (int j = i + 1; j < n; j++)
        {
            sum -= A[i * n + j] * X[j];
        }

        X[i] = sum / A[i * n + i];
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
}


/*
    Calculate infinity norm of residual:

        r = AX - B

        ||r||inf = max |r_i|
*/
double calculateResidual(
    const vector<double>& A,
    const vector<double>& X,
    const vector<double>& B,
    int n)
{
    double maxResidual = 0.0;

    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;

        for (int j = 0; j < n; j++)
        {
            sum += A[i * n + j] * X[j];
        }

        double residual = fabs(sum - B[i]);

        if (residual > maxResidual)
            maxResidual = residual;
    }

    return maxResidual;
}


/*
    Compare sequential and CUDA solutions.
*/
double compareSolutions(
    const vector<double>& X1,
    const vector<double>& X2)
{
    double maxDifference = 0.0;

    for (size_t i = 0; i < X1.size(); i++)
    {
        double difference = fabs(X1[i] - X2[i]);

        if (difference > maxDifference)
            maxDifference = difference;
    }

    return maxDifference;
}


int main()
{
    cout << fixed << setprecision(6);

    /*
        Required problem sizes.
    */
    vector<int> sizes = {1000, 2000};

    /*
        Thread configurations.

        CUDA maximum threads per block is normally 1024.
        Therefore 32x32 = 1024 threads.

        We use square blocks:
            8x8   = 64 threads
            16x16 = 256 threads
            32x32 = 1024 threads

        The "threads" column in the CSV represents the
        dimension of the 2D block.
    */
    vector<int> threadConfigs = {8, 16, 32};

    const int RUNS = 5;

    /*
        CSV output.
    */
    ofstream csv("q2_results.csv");

    csv << "Problem,Method,n,Threads,Blocks,Run,Time_ms,Average_ms,Speedup,Residual,MaxSolutionDifference\n";

    for (int n : sizes)
    {
        cout << "\n============================================================\n";
        cout << "Q2 - GAUSSIAN ELIMINATION\n";
        cout << "n = " << n << "\n";
        cout << "============================================================\n";

        /*
            Generate one fixed matrix for this n.

            This is important because sequential and parallel
            implementations must solve exactly the same system.
        */
        vector<double> A;
        vector<double> B;

        generateSystem(A, B, n);

        /*
            ----------------------------------------------------
            SEQUENTIAL
            ----------------------------------------------------
        */

        vector<double> sequentialX;

        vector<double> sequentialTimes;

        cout << "\nSequential Gaussian Elimination\n";

        for (int run = 1; run <= RUNS; run++)
        {
            double start = getTime();

            gaussianSequential(
                A,
                B,
                sequentialX,
                n);

            double end = getTime();

            double elapsed = end - start;

            sequentialTimes.push_back(elapsed);

            cout << "Run " << run
                 << ": " << elapsed
                 << " ms\n";
        }

        double sequentialAverage = 0.0;

        for (double t : sequentialTimes)
            sequentialAverage += t;

        sequentialAverage /= RUNS;

        double sequentialResidual =
            calculateResidual(
                A,
                sequentialX,
                B,
                n);

        cout << "Sequential Average: "
             << sequentialAverage
             << " ms\n";

        cout << "Sequential Residual: "
             << sequentialResidual
             << "\n";

        /*
            Write sequential results.
        */
        for (int run = 0; run < RUNS; run++)
        {
            csv << "Q2,Sequential,"
                << n << ","
                << 0 << ","
                << 0 << ","
                << run + 1 << ","
                << sequentialTimes[run] << ","
                << sequentialAverage << ","
                << 1.0 << ","
                << sequentialResidual << ","
                << 0.0 << "\n";
        }

        /*
            ----------------------------------------------------
            CUDA CONFIGURATIONS
            ----------------------------------------------------
        */

        for (int threads : threadConfigs)
        {
            cout << "\n------------------------------------------------------------\n";
            cout << "CUDA Configuration\n";
            cout << "Block = "
                 << threads
                 << " x "
                 << threads
                 << " = "
                 << threads * threads
                 << " threads\n";

            vector<double> cudaTimes;

            vector<double> cudaX;

            /*
                Number of logical 1D blocks used for rows.

                For reporting, we calculate approximately:
                    ceil(n / threads)
            */
            int blocks =
                (n + threads - 1) / threads;

            for (int run = 1; run <= RUNS; run++)
            {
                CUDA_CHECK(cudaDeviceSynchronize());

                double start = getTime();

                gaussianCUDA(
                    A,
                    B,
                    cudaX,
                    n,
                    threads);

                CUDA_CHECK(cudaDeviceSynchronize());

                double end = getTime();

                double elapsed = end - start;

                cudaTimes.push_back(elapsed);

                cout << "Run " << run
                     << ": " << elapsed
                     << " ms\n";
            }

            double cudaAverage = 0.0;

            for (double t : cudaTimes)
                cudaAverage += t;

            cudaAverage /= RUNS;

            double speedup =
                sequentialAverage / cudaAverage;

            double cudaResidual =
                calculateResidual(
                    A,
                    cudaX,
                    B,
                    n);

            double maxDifference =
                compareSolutions(
                    sequentialX,
                    cudaX);

            cout << "Average: "
                 << cudaAverage
                 << " ms\n";

            cout << "Speedup: "
                 << speedup
                 << "x\n";

            cout << "CUDA Residual: "
                 << cudaResidual
                 << "\n";

            cout << "Max Difference from Sequential: "
                 << maxDifference
                 << "\n";

            if (cudaResidual < EPSILON)
                cout << "Residual Check: PASS\n";
            else
                cout << "Residual Check: FAIL\n";

            if (maxDifference < 1e-6)
                cout << "Sequential/CUDA Check: PASS\n";
            else
                cout << "Sequential/CUDA Check: CHECK\n";

            /*
                Write CUDA results to CSV.
            */
            for (int run = 0; run < RUNS; run++)
            {
                csv << "Q2,CUDA,"
                    << n << ","
                    << threads << ","
                    << blocks << ","
                    << run + 1 << ","
                    << cudaTimes[run] << ","
                    << cudaAverage << ","
                    << speedup << ","
                    << cudaResidual << ","
                    << maxDifference << "\n";
            }
        }
    }

    csv.close();

    cout << "\n============================================================\n";
    cout << "EXPERIMENT COMPLETED\n";
    cout << "Results saved to q2_results.csv\n";
    cout << "============================================================\n";

    return 0;
}
