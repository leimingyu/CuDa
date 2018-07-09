/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
	inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

void profileCopies(float        *h_a, 
		float        *h_b, 
		float        *d, 
		unsigned int  n,
		const char         *desc)
{
	printf("\n%s transfers\n", desc);

	unsigned int bytes = n * sizeof(float);

	// events for timing
	cudaEvent_t startEvent, stopEvent; 

	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	checkCuda( cudaEventRecord(startEvent, 0) );
	checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );

	float time;
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

	checkCuda( cudaEventRecord(startEvent, 0) );
	checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );

	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

	for (int i = 0; i < n; ++i) {
		if (h_a[i] != h_b[i]) {
			printf("*** %s transfers failed ***", desc);
			break;
		}
	}

	// clean up events
	checkCuda( cudaEventDestroy(startEvent) );
	checkCuda( cudaEventDestroy(stopEvent) );
}

int main(int argc, char **argv)
{
	int N = 1024 * 1024;   // 1M floats = 4MB

	int devID = 0;                                                              
	if(argc == 2) {                                                             
		devID = atoi(argv[1]);                                                  
	}                                                                           

	if(argc == 3) {                                                             
		Tsize = atoi(argv[2]);                                                  
	}                                                                           

	printf("select device : %d\n", devID);                                      
	cudaSetDevice(devID);   


	//// output device info and transfer size
	cudaDeviceProp prop;
	checkCuda( cudaGetDeviceProperties(&prop, devID) );
	printf("\nDevice: %s\n", prop.name);

	const unsigned int bytes = N * sizeof(float);

	// host arrays
	float *h_a, *h_b, *h_c;   

	// device array
	float *d_a, *d_b, *d_c;

	// allocate and initialize
	h_a = (float*)malloc(bytes);
	h_b = (float*)malloc(bytes);
	h_c = (float*)malloc(bytes);

	checkCuda( cudaMalloc((void**)&d_a, bytes) );
	checkCuda( cudaMalloc((void**)&d_b, bytes) );
	checkCuda( cudaMalloc((void**)&d_c, bytes) );


	for (int i = 0; i < N; ++i) {
		h_a[i] = i;
		h_b[i] = i;
	}

	// copy data to device
	checkCuda( cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice) );
	checkCuda( cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice) );

	int nBlocks = (N + 1023) / 1024;

	dim3 grid(nBlocks, 1, 1);
	dim3 blck(1024, 1, 1);

	//kern_vecAdd <<< grid, blck >>> (d_a, d_b, d_c);




	printf("\n");

	// cleanup
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
