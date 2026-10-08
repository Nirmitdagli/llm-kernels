// Tiled GEMM in CUDA C++, optimized step by step and measured against cuBLAS:
//   1. naive  2. shared-memory tiling  3. 2D register blocking  4. float4 vectorized loads
// C = A x B, all N x N, row-major, fp32.
//
// Build and run (Colab T4):
//   nvcc -O3 -arch=sm_75 gemm.cu -lcublas -o gemm && ./gemm
// (sm_75 = T4. Use sm_80 for A100, sm_89 for L4.)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Kernel 1: naive. One thread computes one C[row][col].
// Every multiply reads A and B straight from global memory (HBM):
// for N = 4096 that is 2 x 4096 global loads per output. Very low reuse.
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x walks columns -> B and C loads are coalesced
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: shared-memory tiling.
// A block of TILE x TILE threads computes a TILE x TILE patch of C.
// It walks along K in steps of TILE: each step loads one tile of A and one
// tile of B into shared memory (one element per thread), syncs, then every
// thread reuses those 2*TILE values from fast on-chip memory.
// Global loads drop by a factor of TILE.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void gemm_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        // Cooperative load: each thread brings in one element of each tile.
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();                     // 1st sync: tiles fully loaded before anyone reads

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[ty][k] * Bs[k][tx];    // all reads hit shared memory
        __syncthreads();                     // 2nd sync: nobody overwrites a tile still being read
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 3: 2D register blocking (block tile 64x64, thread tile 4x4).
// Tiling alone still does 2 shared-memory loads per multiply-add.
// Now each thread computes a 4x4 patch of C, so per k step it loads
// 4 values of A + 4 of B into REGISTERS and does 16 multiply-adds:
// 8 loads per 16 FMAs instead of 2 per 1. 256 threads per block.
// Requires N to be a multiple of 64 (true for all sizes we benchmark).
// ---------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, TM = 4, TN = 4;          // block tile, thread tile
constexpr int THREADS = (BM / TM) * (BN / TN);           // 16 x 16 = 256

template <int BK>
__global__ void gemm_regblock(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN);                     // 0..15: which 4-row strip of the tile
    const int tCol = tid % (BN / TN);                     // 0..15: which 4-col strip of the tile
    const float* Ablk = A + blockIdx.y * BM * N;          // top-left of this block's rows of A
    const float* Bblk = B + blockIdx.x * BN;              // top-left of this block's cols of B
    float* Cblk = C + blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM][TN] = {};
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Cooperative load: BM*BK elements of A and BK*BN of B, spread over 256 threads.
        for (int i = tid; i < BM * BK; i += THREADS)
            As[i / BK][i % BK] = Ablk[(i / BK) * N + k0 + i % BK];
        for (int i = tid; i < BK * BN; i += THREADS)
            Bs[i / BN][i % BN] = Bblk[(k0 + i / BN) * N + i % BN];   // consecutive threads -> consecutive columns: coalesced
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) regA[m] = As[tRow * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; ++n) regB[n] = Bs[k][tCol * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += regA[m] * regB[n];       // 16 FMAs from 8 register values
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
            Cblk[(tRow * TM + m) * N + tCol * TN + n] = acc[m][n];
}

// ---------------------------------------------------------------------------
// Kernel 4: register blocking + float4 VECTORIZED loads and stores.
// Same 64x64 block tile and 4x4 thread tile, BK = 16, 256 threads.
// - Global loads: each thread reads ONE float4 (16 bytes) of A and ONE of B per
//   k step: 4x fewer load instructions than scalar loads.
// - A is stored TRANSPOSED in shared memory (AsT[k][m]) so a thread's 4 A values
//   for a given k are contiguous and can be read as one float4.
// - C is written back with float4 stores.
// Requires N % 64 == 0 (also guarantees 16-byte alignment of every float4).
// ---------------------------------------------------------------------------
constexpr int VBK = 16;

__global__ void gemm_vec4(const float* A, const float* B, float* C, int N) {
    __shared__ __align__(16) float AsT[VBK][BM];          // transposed A tile
    __shared__ __align__(16) float Bs[VBK][BN];

    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN);
    const int tCol = tid % (BN / TN);
    const float* Ablk = A + blockIdx.y * BM * N;
    const float* Bblk = B + blockIdx.x * BN;
    float* Cblk = C + blockIdx.y * BM * N + blockIdx.x * BN;

    // Which float4 this thread loads: A tile is 64 rows x 16 cols = 64 x 4 float4s,
    // B tile is 16 rows x 64 cols = 16 x 16 float4s. Both are exactly 256 float4s.
    const int aRow = tid / (VBK / 4), aCol = (tid % (VBK / 4)) * 4;
    const int bRow = tid / (BN / 4),  bCol = (tid % (BN / 4)) * 4;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < N; k0 += VBK) {
        float4 a = *reinterpret_cast<const float4*>(&Ablk[aRow * N + k0 + aCol]);
        AsT[aCol + 0][aRow] = a.x;                         // scatter into the transposed tile
        AsT[aCol + 1][aRow] = a.y;
        AsT[aCol + 2][aRow] = a.z;
        AsT[aCol + 3][aRow] = a.w;
        *reinterpret_cast<float4*>(&Bs[bRow][bCol]) =
            *reinterpret_cast<const float4*>(&Bblk[(k0 + bRow) * N + bCol]);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < VBK; ++k) {
            float4 ra = *reinterpret_cast<const float4*>(&AsT[k][tRow * TM]);  // 4 A values, 1 load
            float4 rb = *reinterpret_cast<const float4*>(&Bs[k][tCol * TN]);   // 4 B values, 1 load
            float av[4] = {ra.x, ra.y, ra.z, ra.w};
            float bv[4] = {rb.x, rb.y, rb.z, rb.w};
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += av[m] * bv[n];
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; ++m)
        *reinterpret_cast<float4*>(&Cblk[(tRow * TM + m) * N + tCol * TN]) =
            make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
template <typename F>
float time_ms(F launch, int iters = 10) {
    launch();                                // warm-up (first launch pays setup costs)
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    // "Effective GB/s" = minimum bytes a GEMM must move (read A and B, write C) / time.
    // It shows how far each kernel is from being limited by memory bandwidth.
    printf("| N | Kernel | ms | GFLOP/s | Effective GB/s | %% of cuBLAS | max error vs cuBLAS |\n");
    printf("| --- | --- | --- | --- | --- | --- | --- |\n");

    for (int N : {512, 1024, 2048, 4096}) {
        size_t bytes = (size_t)N * N * sizeof(float);
        std::vector<float> hA(N * N), hB(N * N), ref(N * N), out(N * N);
        for (auto& v : hA) v = (rand() % 100) / 100.0f - 0.5f;
        for (auto& v : hB) v = (rand() % 100) / 100.0f - 0.5f;

        float *A, *B, *C;
        CHECK(cudaMalloc(&A, bytes)); CHECK(cudaMalloc(&B, bytes)); CHECK(cudaMalloc(&C, bytes));
        CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));
        double flops = 2.0 * N * N * N;      // one multiply + one add per inner step

        // cuBLAS is column-major. Row-major C = A*B equals column-major C^T = B^T * A^T,
        // so we pass B first, then A, and get row-major C back.
        float alpha = 1.0f, beta = 0.0f;
        auto run_cublas = [&] {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N);
        };
        float ms_cublas = time_ms(run_cublas);
        CHECK(cudaMemcpy(ref.data(), C, bytes, cudaMemcpyDeviceToHost));

        dim3 blk16(16, 16), grid16((N + 15) / 16, (N + 15) / 16);
        dim3 blk32(32, 32), grid32((N + 31) / 32, (N + 31) / 32);

        struct Case { const char* name; float ms; float err; };
        std::vector<Case> cases;

        float ms = time_ms([&] { gemm_naive<<<grid16, blk16>>>(A, B, C, N); });
        CHECK(cudaMemcpy(out.data(), C, bytes, cudaMemcpyDeviceToHost));
        cases.push_back({"naive", ms, max_abs_diff(out, ref)});

        ms = time_ms([&] { gemm_tiled<16><<<grid16, blk16>>>(A, B, C, N); });
        CHECK(cudaMemcpy(out.data(), C, bytes, cudaMemcpyDeviceToHost));
        cases.push_back({"tiled 16", ms, max_abs_diff(out, ref)});

        ms = time_ms([&] { gemm_tiled<32><<<grid32, blk32>>>(A, B, C, N); });
        CHECK(cudaMemcpy(out.data(), C, bytes, cudaMemcpyDeviceToHost));
        cases.push_back({"tiled 32", ms, max_abs_diff(out, ref)});

        dim3 blkRB(THREADS), gridRB(N / BN, N / BM);    // N is a multiple of 64 for every size we run
        ms = time_ms([&] { gemm_regblock<8><<<gridRB, blkRB>>>(A, B, C, N); });
        CHECK(cudaMemcpy(out.data(), C, bytes, cudaMemcpyDeviceToHost));
        cases.push_back({"regblock 4x4", ms, max_abs_diff(out, ref)});

        ms = time_ms([&] { gemm_vec4<<<gridRB, blkRB>>>(A, B, C, N); });
        CHECK(cudaMemcpy(out.data(), C, bytes, cudaMemcpyDeviceToHost));
        cases.push_back({"vec4 + regblock", ms, max_abs_diff(out, ref)});

        cases.push_back({"cuBLAS", ms_cublas, 0.0f});

        for (auto& c : cases)
            printf("| %d | %s | %.3f | %.0f | %.0f | %.1f%% | %.2e |\n", N, c.name, c.ms,
                   flops / (c.ms * 1e-3) / 1e9, 3.0 * bytes / (c.ms * 1e-3) / 1e9,
                   100.0 * ms_cublas / c.ms, c.err);

        cudaFree(A); cudaFree(B); cudaFree(C);
    }
    cublasDestroy(handle);
    return 0;
}
