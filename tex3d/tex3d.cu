#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>


// data type: float
// 3 channels
// cudaReadModeElementType:	Read texture as specified element type
// cudaReadModeNormalizedFloat: Read texture as normalized float
texture<float, 3, cudaReadModeElementType> tex3d;


__global__ void test_kernel(const int w, const int h, const int d) 
{
    float data;

    //data = tex3D(tex3d, 0.f, 0.f, (float) threadIdx.x);
	//printf("thread %d, data %f\n", threadIdx.x, data);

    data = tex3D(tex3d, 0.f, 2.f, (float) threadIdx.x);
	printf("thread %d, data %f\n", threadIdx.x, data);
}

//----------------------------------------------------------------------------//
// main
//----------------------------------------------------------------------------//
int main(void)
{
	int w = 2;
	int h = 3;
	int d = 4;

	//-----------//
	// 3d array on the host 
	//-----------//
	float *h_array = NULL;
	h_array = (float*) malloc(w * h * d * sizeof(float));

	printf("\nInput array:\n\n");

	for (int i=0; i<w; i++) {
		for (int j=0; j<h; j++) {
			for (int k=0; k<d; k++) {
				h_array[k*w*h + j*w + i] = (float)(i + j + k);
				printf("%12.6f ", (float)(i + j + k));
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");

	//-----------//
	// 3d array on device
	//-----------//
	// set up the cuda array
	cudaArray *d_array= NULL;
	cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc<float>();
	cudaExtent const array_dim = {w, h, d};
	checkCudaErrors(cudaMalloc3DArray(&d_array, &chanDesc, array_dim));


	// paramters
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.extent   = array_dim;
	copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_array, 
			array_dim.width*sizeof(float), array_dim.width, array_dim.height);
	copyParams.dstArray = d_array;
	copyParams.kind     = cudaMemcpyHostToDevice;
	
	//---------------//
	// copy data from host to device
	//---------------//
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	//---------------//
	// binding array to texture
	//---------------//
	// set texture parameters
	// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4Xfram8Ds
	tex3d.normalized = false;
	tex3d.filterMode = cudaFilterModePoint;
	tex3d.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
	tex3d.addressMode[1] = cudaAddressModeWrap;
	tex3d.addressMode[2] = cudaAddressModeWrap;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex3d, d_array, chanDesc));

	test_kernel <<< 1, 4 >>> (w,h,d);	

	if(h_array != NULL) free(h_array);

	if(d_array != NULL) cudaFree(d_array);

	cudaDeviceReset();

	return 0;
}
