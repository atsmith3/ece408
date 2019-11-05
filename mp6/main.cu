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

__global__ void cast_float_to_uint8(float* input, uint8_t* output, int width, int height, int channels) {


}

__global__ void cast_uint8_to_float(uint8_t* input, float* output, int width, int height, int channels) {


}

__global__ void convert_to_greyscale(uint8_t* input, uint8_t* gs, int width, int height, int channels) {


}

__global__ void histogram(uint8_t* gs, uint32_t* hist) {


}

__global__ void scan(uint32_t* hist, uint32_t* cdf, int len) {


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
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  //@@ insert code here

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
