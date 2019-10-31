#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 512

#define _check(stmt)                                                      \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("Failed to run stmt ", #stmt);                               \
      printf("Got CUDA error ...  ", cudaGetErrorString(err));            \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < len) {
    out[idx] = in1[idx] + in2[idx];
  }
}

float* importData(char* fname, int* len) {
  FILE* infile = fopen(fname, "r");
  fscanf(infile, "%d\n", len);
  float* data = (float*)malloc((*len)*sizeof(float));
  for(int i = 0; i < *len; i++) {
    fscanf(infile, "%f\n", &data[i]);
  }
  return data;
}

void printOutput(float* data, int len) {
  printf("[");
  for(int i = 0; i < len; i++) {
    if(i == len-1) {
      printf("%f]\n", data[i]);
    }
    else {
      printf("%f, ", data[i]);
    }
  }
}


int main(int argc, char **argv) {
  if(argc != 3) {
    printf("Usage: ./scan <input_data_a> <input_data_b>\n");
    return -1;
  }

  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  hostInput1 = importData(argv[1], &inputLength);
  hostInput2 = importData(argv[2], &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  //Alloc inputLength sized array of floats on Device
  _check(cudaMalloc(&deviceInput1, inputLength*(sizeof(float))));
  _check(cudaMalloc(&deviceInput2, inputLength*(sizeof(float))));
  _check(cudaMalloc(&deviceOutput, inputLength*(sizeof(float))));

  //Copy from Host to Device Memory
  _check(cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(float), cudaMemcpyHostToDevice));
  _check(cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(float), cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  int blockSize = 256;
  int gridSize = ceil((float)inputLength/blockSize);

  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();

  _check(cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost));

  printOutput(hostOutput, inputLength);

  _check(cudaFree(deviceInput1));
  _check(cudaFree(deviceInput2));
  _check(cudaFree(deviceOutput));

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
