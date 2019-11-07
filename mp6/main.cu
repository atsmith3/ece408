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
  int pos = blockIdx.x*blockDim.x + threadIdx.x;
  if(pos < width*height) {
    uint8_t r = input[channels*pos];
    uint8_t b = input[channels*pos+1];
    uint8_t g = input[channels*pos+2];
    gs[pos] = (uint8_t) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histogram(uint8_t* gs, uint32_t* hist, int width, int height) {
  __shared__ uint32_t local_hist[HISTOGRAM_LENGTH];
  int pos = blockDim.x*blockIdx.x + threadIdx.x;
  if(threadIdx.x < HISTOGRAM_LENGTH) {
    local_hist[threadIdx.x] = 0;
  }

  // Compute local histogram
  __syncthreads();
  if(pos < width*height) {
    atomicAdd(&(local_hist[gs[pos]]), 1);
  }
  
  // Add local to global histogram
  __syncthreads();
  if(threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(hist[threadIdx.x]), local_hist[threadIdx.x]);
  }
}

__global__ void scan(uint32_t* hist, float* cdf, int width, int height) {
  __shared__ float ls[HISTOGRAM_LENGTH];

  int stride = 0x1;
  int i = threadIdx.x;
  int ind = threadIdx.x + blockIdx.x*blockDim.x;
  float temp;

  if(ind < HISTOGRAM_LENGTH){
    ls[i] = (float)hist[ind];
  }
  else {
    ls[i] = 0.0;
  }
  
  for(stride = 0x1; stride < blockDim.x; stride = stride << 1) {
    __syncthreads();
    if(i >= stride) {
      temp = ls[i] + ls[i-stride];
    }
    __syncthreads();
    if(i >= stride) {
      ls[i] = temp;
    }
  }

  __syncthreads();
  if(ind < HISTOGRAM_LENGTH) {
    cdf[ind] = ls[i]/((float)(width*height));
  } 

}

__global__ void equalize(float* cdf, uint8_t* input, uint8_t* output, int width, int height, int channels) {
  int pos = blockIdx.x*blockDim.x + threadIdx.x;
  if(pos < width*height*channels) {
    output[pos] = (uint8_t)min(max((255.0*(cdf[input[pos]] - cdf[0])/(1.0 - cdf[0])), 0.0), 255.0);
  }
}

void printcdf(float* cdf) {
  printf("cdf: [");
  for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
    if(i == HISTOGRAM_LENGTH - 1) {
      printf("%.2f]\n", cdf[i]);
    }
    else {
      printf("%.2f, ", cdf[i]);
    }
  }
}

void printhist(uint32_t* hist) {
  printf("hist: [");
  for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
    if(i == HISTOGRAM_LENGTH - 1) {
      printf("%u]\n", hist[i]);
    }
    else {
      printf("%u, ", hist[i]);
    }
  }
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
  uint32_t* deviceHist;
  uint32_t* hostHist;
  float* deviceCDF;
  float* hostCDF;
  uint8_t* deviceEqImage;

  cudaMalloc((void**)&deviceInputImage, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void**)&deviceOutputImage, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void**)&deviceUcharImage, imageWidth*imageHeight*imageChannels*sizeof(uint8_t));
  cudaMalloc((void**)&deviceGSImage, imageWidth*imageHeight*sizeof(uint8_t));
  cudaMalloc((void**)&deviceHist, HISTOGRAM_LENGTH*sizeof(uint32_t));
  cudaMalloc((void**)&deviceCDF, HISTOGRAM_LENGTH*sizeof(float));
  cudaMalloc((void**)&deviceEqImage, imageWidth*imageHeight*imageChannels*sizeof(uint8_t));

  cudaMemset((void*)deviceHist, 0, HISTOGRAM_LENGTH*sizeof(uint32_t));
  cudaMemset((void*)deviceCDF, 0, HISTOGRAM_LENGTH*sizeof(float));

  hostHist = (uint32_t*)malloc(HISTOGRAM_LENGTH*sizeof(uint32_t));
  hostCDF = (float*)malloc(HISTOGRAM_LENGTH*sizeof(float));

  cudaMemcpy(deviceInputImage, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid;
  dim3 block;

  // Convert float to uint8_t
  grid = dim3(ceil((float)(imageWidth*imageHeight*imageChannels)/512), 1, 1);
  block = dim3(512, 1, 1);
  f2u<<<grid, block>>>(deviceInputImage, deviceUcharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  // Convert to Greyscale:
  grid = dim3(ceil((float)(imageWidth*imageHeight)/512), 1, 1);
  block = dim3(512, 1, 1);
  conv_gs<<<grid, block>>>(deviceUcharImage, deviceGSImage, imageWidth, imageHeight, imageChannels);

  // Calculate Histogram:
  grid = dim3(ceil((float)(imageWidth*imageHeight)/512), 1, 1);
  block = dim3(512, 1, 1);
  histogram<<<grid, block>>>(deviceGSImage, deviceHist, imageWidth, imageHeight);
  cudaMemcpy(hostHist, deviceHist, HISTOGRAM_LENGTH*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  printhist(hostHist);

  // Calculate CDF:
  grid = dim3(1, 1, 1);
  block = dim3(HISTOGRAM_LENGTH, 1, 1);
  scan<<<grid, block>>>(deviceHist, deviceCDF, imageWidth, imageHeight);
  cudaMemcpy(hostCDF, deviceCDF, HISTOGRAM_LENGTH*sizeof(float), cudaMemcpyDeviceToHost);
  printcdf(hostCDF);

  // Equalize Image:
  grid = dim3(ceil((float)(imageWidth*imageHeight*imageChannels)/512), 1, 1);
  block = dim3(512, 1, 1);
  equalize<<<grid, block>>>(deviceCDF, deviceUcharImage, deviceEqImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  // Convert uint8_t to float
  grid = dim3(ceil((float)(imageWidth*imageHeight*imageChannels)/512), 1, 1);
  block = dim3(512, 1, 1);
  u2f<<<grid, block>>>(deviceEqImage, deviceOutputImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImage, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
