#include "cuda_fp16.h"

#define BLOCK 128

#ifndef ZERO_FLOAT
    #define ZERO_FLOAT 2.2e-20
#endif

#ifndef ZERO_DOUBLE
    #define ZERO_DOUBLE 1.4e-40
#endif

#ifndef ZERO_HALF
    #define ZERO_HALF 4.166e-05
#endif

__device__ __forceinline__ void axpy__(const double a, const double b, double &c) {
    c = __fma_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, float &c) {
    c = __fmaf_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const double a, const double b, float &c) {
    c = __fmaf_rn(__double2float_rn(a), __double2float_rn(b), c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, __half &c) {
    c = __hfma(__float2half(a), __float2half(b), c);
}

__device__ unsigned long long errors = 0;

template<const uint32 THRESHOLD_UINT32>
__device__ void check_bit_error(const __half &lhs, const float &rhs) {
	const uint32 lhs_data = __half2uint_rn(lhs);
	const uint32 rhs_data = __float_as_uint(rhs);
	uint32 sub_res;
	if (lhs_data > rhs_data) {
		sub_res = lhs_data - rhs_data;
	} else {
		sub_res = rhs_data - lhs_data;
	}

	if (sub_res > THRESHOLD_UINT32) {
		atomicAdd(&errors, 1);
	}
}

template<const uint32 THRESHOLD_UINT32>
__device__ void check_bit_error(const float &lhs, const float &rhs) {
	float diff = fabs(lhs - rhs);
	if (diff > ZERO_FLOAT) {
		atomicAdd(&errors, 1);
	}
}

template<const uint32 THRESHOLD, const uint32 COUNT, typename real_t, typename half_t>
__global__ void matrix_mult_dmr_kernel(const real_t *A, const real_t *B, int M, int N, int K, real_t *C, half_t *C_h) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < m && col < n) {
        register real_t acc_real_t = 0.0;
	    register half_t acc_half_t = 0.0;

        #pragma unroll COUNT
        for (int i = 0; i < K; i++) {
            axpy__(A[row * M + i], B[col * N + i], acc_real_t);
            axpy__(A[row * M + i], B[col * N + i], acc_half_t);

            if ((i % COUNT) == 0) {
                check_bit_error<THRESHOLD>(acc_half_t, acc_real_t);
                acc_half_t = half_t(acc_real_t);
            }
        }

        C[row * M + col] = acc_real_t;
        C_h[row * M + col] = acc_half_t;
    }

}

template<const uint32 THRESHOLD, const uint32 COUNT, typename real_t, typename half_t>
void matrix_mult_dmr(const real_t *A, const real_t *B, int M, int N, int K, real_t *C, half_t *C_h) {
    dim3 threads(BLOCK, BLOCK);
	dim3 grid(ceil(float(M)/BLOCK), ceil(float(N)/BLOCK));
    matrix_mult_dmr_kernel<<<grid,threads>>>(A, B, M, N, K, C, C_h);
}