#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define _check(stmt)                                                      \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("Failed to run stmt ", #stmt);                               \
      printf("Got CUDA error ...  ", cudaGetErrorString(err));            \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
  int element = blockIdx.x * blockDim.x + threadIdx.x;
  
  int row = element/numCColumns;
  int column = element%numCColumns;
  
  if(element < numCRows*numCColumns) {
    C[element] = 0;
    for(int i = 0; i < numAColumns; i++) {
      C[element] += A[row*numAColumns + i] * B[i*numBColumns + column];
    } 
  }
}

float* importMatrix(char* fname, int* row, int* col) {
  FILE* infile = fopen(fname, "r");
  fscanf(infile, "%d %d", row, col);
  float* data = (float*)malloc((*row)*(*col)*sizeof(float));
  for(int i = 0; i < (*row); i++) {
    for(int j = 0; j < (*col); j++) {
      if(j = *col - 1) {
        fscanf(infile, "%f\n", &data[i*(*col) + j]);
      }
      else {
        fscanf(infile, "%f ", &data[i*(*col) + j]);
      }
    }
  }
  return data;
}

void printOutput(float* data, int row, int col) {
  printf("[");
  for(int i = 0; i < row; i++) {
    for(int j = 0; j < row; j++) {
      if(i == len-1) {
        printf(".1%f]\n", data[i]);
      }
      else {
        printf(".1%f, ", data[i]);
      }
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

  hostInput1 = importMatrix(argv[1], &inputLength);
  hostInput2 = importMatrix(argv[2], &inputLength);
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
