#include "device_vector.h"
#include "kernels.h"
#include <cassert>
#include <vector>
#include <iostream>

#define HALF_ROUND_STYLE 1  // 1: nearest, -1: truncate (fastest, default)
#include "half.hpp"
using half_float::half;
using namespace half_float::literal;


#define CHECK_BLOCK 100
#define THRESHOLD 1

typedef double real_t;
typedef half_float::half half_t_host;

int main(int argc, char **argv) {

	int m;
	int n;
	int k;
	m = n = k = 4096;
	int lda = m;
	int ldb = n;
	int ldc = k;

	real_t alpha = 1;
	real_t beta = 1;

	const std::vector<real_t> zero_vector(m * k, 0.0);
	const std::vector<half_t_host> zero_vector_half(m * k, 0.0);

	std::vector<real_t> host_a(m * k, alpha);
	std::vector<real_t> host_b(k * n, beta);
	std::vector<real_t> host_c(m * n, 0.0);
	std::vector<half_t_host> host_c_half(m * n, 0.0);

	rad::DeviceVector<real_t> device_a(host_a), device_b(host_b), device_c(host_c);
	rad::DeviceVector<half_t_host> device_c_half(host_c_half);

	matrix_mult_dmr<THRESHOLD, CHECK_BLOCK>(device_a.data(), device_b.data(), m, n, k, device_c.data(), device_c_half.data());

	host_c = device_c.to_vector();
	host_c_inc = device_c_inc.to_vector();

    std::cout << "FLOAT" << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << host_c[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "HALF" << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << host_c_inc[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
    
	return 0;
}