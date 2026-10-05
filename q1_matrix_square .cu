#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) \
             << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

const int RUNS = 5;
const float EPSILON = 0.01f;

/* gettimeofday() is required by the assignment */
double getTimeMs() {
    timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec * 1000.0 + t.tv_usec / 1000.0;
}

/* Generate a row-major matrix */
void generateMatrix(vector<float>& A, int n) {
    srand(42);

    for (long long i = 0; i < (long long)n * n; i++)
        A[i] = (rand() % 21) - 10;
}

/* Sequential reference implementation */
void cpuMatrixSquare(const float* A, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];

            for (int j = 0; j < n; j++) {
                C[i * n + j] += a * A[k * n + j];
            }
        }
    }
}

/* Basic CUDA matrix multiplication */
__global__ void basicMatrixSquare(
    const float* A,
    float* C,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;

        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * A[k * n + col];

        C[row * n + col] = sum;
    }
}

/*
   Shared-memory tiled CUDA implementation.

   A^2 = A * A

   tileA contains a row portion of A.
   tileB contains a column portion of A.
*/
template<int TILE>
__global__ void tiledMatrixSquare(
    const float* A,
    float* C,
    int n
) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int numberOfTiles = (n + TILE - 1) / TILE;

    for (int tile = 0; tile < numberOfTiles; tile++) {
        int aCol = tile * TILE + tx;
        int bRow = tile * TILE + ty;

        if (row < n && aCol < n)
            tileA[ty][tx] = A[row * n + aCol];
        else
            tileA[ty][tx] = 0.0f;

        if (bRow < n && col < n)
            tileB[ty][tx] = A[bRow * n + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += tileA[ty][k] * tileB[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

/* Launch basic CUDA kernel */
void launchBasic(
    const float* d_A,
    float* d_C,
    int n,
    int blockSize
) {
    dim3 block(blockSize, blockSize);
    dim3 grid(
        (n + blockSize - 1) / blockSize,
        (n + blockSize - 1) / blockSize
    );

    basicMatrixSquare<<<grid, block>>>(d_A, d_C, n);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

/* Launch tiled CUDA kernel */
void launchTiled(
    const float* d_A,
    float* d_C,
    int n,
    int blockSize
) {
    dim3 block(blockSize, blockSize);
    dim3 grid(
        (n + blockSize - 1) / blockSize,
        (n + blockSize - 1) / blockSize
    );

    if (blockSize == 8) {
        tiledMatrixSquare<8><<<grid, block>>>(d_A, d_C, n);
    }
    else if (blockSize == 16) {
        tiledMatrixSquare<16><<<grid, block>>>(d_A, d_C, n);
    }
    else {
        tiledMatrixSquare<32><<<grid, block>>>(d_A, d_C, n);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

/* Check GPU result against CPU result */
bool verifyResult(
    const vector<float>& reference,
    const vector<float>& result,
    int n
) {
    long long total = (long long)n * n;

    for (long long i = 0; i < total; i++) {
        float difference = fabs(reference[i] - result[i]);

        if (difference > EPSILON) {
            cout << "Verification failed at index " << i
                 << " | CPU = " << reference[i]
                 << " | GPU = " << result[i]
                 << " | Difference = " << difference << endl;
            return false;
        }
    }

    return true;
}

int main() {
    cout << fixed << setprecision(5);

    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cerr << "No CUDA GPU found." << endl;
        return 1;
    }

    int sizes[] = {1000, 2000};
    int blockSizes[] = {8, 16, 32};

    ofstream csv("q1_results.csv");

    if (!csv.is_open()) {
        cerr << "Could not create q1_results.csv" << endl;
        return 1;
    }

    csv << "Matrix_Size,Implementation,Block_Size,Threads_Per_Block,"
        << "Grid_X,Grid_Y,Run,Time_ms,Average_ms,Speedup,Correctness\n";

    for (int n : sizes) {
        cout << "\n============================================\n";
        cout << "Matrix Size: " << n << " x " << n << endl;
        cout << "============================================\n";

        long long elements = (long long)n * n;
        size_t bytes = elements * sizeof(float);

        vector<float> A(elements);
        vector<float> cpuResult(elements, 0.0f);
        vector<float> gpuResult(elements);

        generateMatrix(A, n);

        /* -----------------------------------------
           Sequential CPU implementation
           ----------------------------------------- */

        cout << "\nSequential CPU\n";

        double cpuTimes[RUNS];
        double cpuTotal = 0.0;

        for (int run = 0; run < RUNS; run++) {
            fill(cpuResult.begin(), cpuResult.end(), 0.0f);

            double start = getTimeMs();

            cpuMatrixSquare(
                A.data(),
                cpuResult.data(),
                n
            );

            double end = getTimeMs();

            cpuTimes[run] = end - start;
            cpuTotal += cpuTimes[run];

            cout << "Run " << run + 1
                 << ": " << cpuTimes[run]
                 << " ms" << endl;
        }

        double cpuAverage = cpuTotal / RUNS;

        cout << "Average: "
             << cpuAverage << " ms" << endl;

        csv << n
            << ",CPU,0,0,0,0,0,"
            << cpuAverage << ","
            << cpuAverage << ",1,PASS\n";

        /* -----------------------------------------
           Allocate GPU memory
           ----------------------------------------- */

        float* d_A = nullptr;
        float* d_C = nullptr;

        CHECK(cudaMalloc(&d_A, bytes));
        CHECK(cudaMalloc(&d_C, bytes));

        CHECK(cudaMemcpy(
            d_A,
            A.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        /* -----------------------------------------
           Different block sizes
           ----------------------------------------- */

        for (int blockSize : blockSizes) {
            int gridX = (n + blockSize - 1) / blockSize;
            int gridY = (n + blockSize - 1) / blockSize;
            int threadsPerBlock = blockSize * blockSize;

            cout << "\n--------------------------------------------\n";
            cout << "Block Size: "
                 << blockSize << " x "
                 << blockSize << endl;

            cout << "Threads per Block: "
                 << threadsPerBlock << endl;

            cout << "Grid Size: "
                 << gridX << " x "
                 << gridY << endl;

            cout << "--------------------------------------------\n";

            /* =========================================
               BASIC CUDA
               ========================================= */

            cout << "\nBasic CUDA\n";

            /* Warm-up run */
            launchBasic(d_A, d_C, n, blockSize);

            double basicTimes[RUNS];
            double basicTotal = 0.0;

            for (int run = 0; run < RUNS; run++) {
                double start = getTimeMs();

                launchBasic(
                    d_A,
                    d_C,
                    n,
                    blockSize
                );

                double end = getTimeMs();

                basicTimes[run] = end - start;
                basicTotal += basicTimes[run];

                cout << "Run " << run + 1
                     << ": " << basicTimes[run]
                     << " ms" << endl;
            }

            double basicAverage = basicTotal / RUNS;
            double basicSpeedup = cpuAverage / basicAverage;

            CHECK(cudaMemcpy(
                gpuResult.data(),
                d_C,
                bytes,
                cudaMemcpyDeviceToHost
            ));

            bool basicCorrect = verifyResult(
                cpuResult,
                gpuResult,
                n
            );

            cout << "Average: "
                 << basicAverage << " ms" << endl;

            cout << "Speedup: "
                 << basicSpeedup << "x" << endl;

            cout << "Correctness: "
                 << (basicCorrect ? "PASS" : "FAIL")
                 << endl;

            for (int run = 0; run < RUNS; run++) {
                csv << n
                    << ",Basic_CUDA,"
                    << blockSize << ","
                    << threadsPerBlock << ","
                    << gridX << ","
                    << gridY << ","
                    << run + 1 << ","
                    << basicTimes[run] << ","
                    << basicAverage << ","
                    << basicSpeedup << ","
                    << (basicCorrect ? "PASS" : "FAIL")
                    << "\n";
            }

            /* =========================================
               TILED CUDA
               ========================================= */

            cout << "\nTiled CUDA\n";

            /* Warm-up run */
            launchTiled(d_A, d_C, n, blockSize);

            double tiledTimes[RUNS];
            double tiledTotal = 0.0;

            for (int run = 0; run < RUNS; run++) {
                double start = getTimeMs();

                launchTiled(
                    d_A,
                    d_C,
                    n,
                    blockSize
                );

                double end = getTimeMs();

                tiledTimes[run] = end - start;
                tiledTotal += tiledTimes[run];

                cout << "Run " << run + 1
                     << ": " << tiledTimes[run]
                     << " ms" << endl;
            }

            double tiledAverage = tiledTotal / RUNS;
            double tiledSpeedup = cpuAverage / tiledAverage;

            CHECK(cudaMemcpy(
                gpuResult.data(),
                d_C,
                bytes,
                cudaMemcpyDeviceToHost
            ));

            bool tiledCorrect = verifyResult(
                cpuResult,
                gpuResult,
                n
            );

            cout << "Average: "
                 << tiledAverage << " ms" << endl;

            cout << "Speedup: "
                 << tiledSpeedup << "x" << endl;

            cout << "Correctness: "
                 << (tiledCorrect ? "PASS" : "FAIL")
                 << endl;

            for (int run = 0; run < RUNS; run++) {
                csv << n
                    << ",Tiled_CUDA,"
                    << blockSize << ","
                    << threadsPerBlock << ","
                    << gridX << ","
                    << gridY << ","
                    << run + 1 << ","
                    << tiledTimes[run] << ","
                    << tiledAverage << ","
                    << tiledSpeedup << ","
                    << (tiledCorrect ? "PASS" : "FAIL")
                    << "\n";
            }
        }

        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_C));
    }

    csv.close();

    cout << "\n============================================\n";
    cout << "Q1 EXPERIMENT COMPLETED\n";
    cout << "Results saved to q1_results.csv\n";
    cout << "============================================\n";

    return 0;
}
