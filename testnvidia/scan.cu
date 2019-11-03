#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "scan_kernels.cuh"
#include "scan.cuh"

#define checkCudaError(o, l) _checkCudaError(o, l, __func__)

int THREADS_PER_BLOCK = 256;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}

// from https://stackoverflow.com/a/12506181
int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

void scan(int *d_out, int *d_in, int length, int* d_sums, int* d_incr, int* d_sums_2, int* d_incr_2)
{
	if (length > ELEMENTS_PER_BLOCK)
	{
		scanLargeDeviceArray(d_out, d_in, length, d_sums, d_incr, d_sums_2, d_incr_2);
	}
	else 
	{
		scanSmallDeviceArray(d_out, d_in, length);
	}
}


void scanLargeDeviceArray(int *d_out, int *d_in, int length, int* d_sums, int* d_incr, int* d_sums_2, int* d_incr_2) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, d_sums, d_incr, d_sums_2, d_incr_2);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, d_sums, d_incr, d_sums_2, d_incr_2);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length) {
	int powerOfTwo = nextPowerOfTwo(length);
	prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, int* d_sums, int* d_incr, int* d_sums_2, int* d_incr_2) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	//int *d_sums, *d_incr;
	//cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	//cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, d_sums_2, d_incr_2, 0, 0);	//hoping this won't be called twice
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);
}