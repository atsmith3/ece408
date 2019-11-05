#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wb.h"

#define HISTOGRAM_LENGTH 256

#define _check(stmt)                                                      \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("Failed to run stmt ", #stmt);                               \
      printf("Got CUDA error ...  ", cudaGetErrorString(err));            \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void f2u(float* input, uint8_t* output, int width, int height, int channels) {
  int pos = blockIdx.x*blockDim.x + threadIdx.x;
  if(pos < width*height*channels) {
    output[pos] = (uint8_t)(255*input[pos]);
  }
}

__global__ void u2f(uint8_t* input, float* output, int width, int height, int channels) {
  int pos = blockIdx.x*blockDim.x + threadIdx.x;
  if(pos < width*height*channels) {
    output[pos] = (float)(input[pos]/255.0);
  }
}

__global__ void conv_gs(uint8_t* input, uint8_t* gs, int width, int height, int channels) {


}

__global__ void histogram(uint8_t* gs, uint32_t* hist, int height, int width, int len) {


}

__global__ void scan(uint32_t* hist, uint32_t* cdf, int len) {


}

__global__ void equalize(uint32_t* hist, int len, uint8_t* image, int height, int width, int channels) {

//  return (uint8_t)min(max((255*(cdf[val] - cdfmin)/(1.0 - cdfmini)), 0), 255.0);

}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  
  args = wbArg_read(argc, argv); /* parse the input arguments */
  
  inputImageFile = wbArg_getInputFile(args, 0);
  
  inputImage = wbImport(inputImageFile);
  hostInputImageData = wbImage_getData(inputImage);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);

  float* deviceInputImage;
  float* deviceOutputImage;
  uint8_t* deviceUcharImage;
  uint8_t* deviceGSImage;

  _check(cudaMalloc((void**)&deviceInputImage, imageWidth*imageHeight*imageChannels*sizeof(float)));
  _check(cudaMalloc((void**)&deviceOutputImage, imageWidth*imageHeight*imageChannels*sizeof(float)));
  _check(cudaMalloc((void**)&deviceUcharImage, imageWidth*imageHeight*imageChannels*sizeof(uint8_t)));
  _check(cudaMalloc((void**)&deviceGSImage, imageWidth*imageHeight*sizeof(uint8_t)));

  _check(cudaMemcpy(deviceInputImage, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice));

  dim3 grid;
  dim3 block;

  // Convert float to uint8_t
  grid = dim3(ceil((float)(imageWidth*imageHeight*imageChannels)/512), 1, 1);
  block = dim3(512, 1, 1);
  f2u<<<grid, block>>>(deviceInputImage, deviceUcharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  // Convert uint8_t to float
  grid = dim3(ceil((float)(imageWidth*imageHeight*imageChannels)/512), 1, 1);
  block = dim3(512, 1, 1);
  u2f<<<grid, block>>>(deviceUcharImage, deviceOutputImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  _check(cudaMemcpy(hostOutputImageData, deviceOutputImage, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost));

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
