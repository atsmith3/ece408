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

__global__ void add(float* arr, float* aux, int len) {
  int j = blockIdx.x-1;
  int i = threadIdx.x + blockIdx.x*BLOCK_SIZE;
  if(i < len && j >= 0) {
    arr[i] = arr[i] + aux[j];
  }
}

__global__ void reduce(float *input, float *output, int len) {
  int id = threadIdx.x;
  int pos = threadIdx.x + blockIdx.x*BLOCK_SIZE;
  if((blockIdx.x+1)*BLOCK_SIZE > len) {
    if(id == (len - blockIdx.x*BLOCK_SIZE - 1)) {
      output[blockIdx.x] = input[pos];
    }
  }
  else {
    output[blockIdx.x] = input[(blockIdx.x+1)*BLOCK_SIZE - 1];
  }
}

__global__ void scan(float *input, float *output, int len) {
  __shared__ float ls[BLOCK_SIZE];

  int stride = 0x1;
  int i = threadIdx.x;
  int ind = threadIdx.x + blockIdx.x*BLOCK_SIZE;
  float temp;
  // Load into the local_scan:
  if(ind < len){
    ls[i] = input[ind];
  }
  else {
    ls[i] = 0.0;
  }
  
  for(stride = 0x1; stride < BLOCK_SIZE; stride = stride << 1) {
    __syncthreads();
    if(i >= stride) {
      temp = ls[i] + ls[i-stride];
    }
    __syncthreads();
    if(i >= stride) {
      ls[i] = temp;
    }
  }
  
  // Copy to output array:
  if(ind < len) {
    output[ind] = ls[i];
  }
}

int main(int argc, char** argv) {
  if(argc != 2) {
    printf("Usage: ./scan <input_data_file>\n");
    return -1;
  }

  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  FILE* infile = fopen(argv[1], "r");
  fscanf(infile, "%d\n", &numElements);

  hostInput = (float*)malloc(numElements * sizeof(float));
  hostOutput = (float *)malloc(numElements * sizeof(float));

  for(int i = 0; i < numElements; i++) {
    int temp;
    fscanf(infile, "%d\n", &temp);
    hostInput[i] = (float)temp;
    printf("%d, ", (int)hostInput[i]);
  }
  printf("\n");

  _check(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  _check(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

  _check(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  _check(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));

  int grid_size = ceil((float)numElements/BLOCK_SIZE);
  dim3 scan_grid(grid_size,1,1);
  dim3 scan_block(BLOCK_SIZE,1,1);
  dim3 aux_grid(ceil((float)grid_size/BLOCK_SIZE),1,1);
  dim3 aux_block(BLOCK_SIZE,1,1);
  dim3 reduce_block(BLOCK_SIZE>>1,1,1);

  // Allocate the auxillary reduction array:
  float* deviceAuxIn;
  float* hostAuxIn;
  float* deviceAuxOut;
  float* hostAuxOut;
  _check(cudaMalloc((void **)&deviceAuxIn, grid_size*sizeof(float)));
  _check(cudaMalloc((void **)&deviceAuxOut, grid_size*sizeof(float)));
  hostAuxIn = (float*)malloc(grid_size*sizeof(float));
  hostAuxIn = (float*)malloc(grid_size*sizeof(float));

  // 1. Scan individual blocks:
  scan<<<scan_grid, scan_block>>>(deviceInput, deviceOutput, numElements);

  // 2. Compute a Reduction on Each Block:
  reduce<<<scan_grid, reduce_block>>>(deviceOutput, deviceAuxIn, numElements);

  _check(cudaMemcpy(hostAuxIn, deviceAuxIn, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < grid_size; i++) {
    printf("%d, ",(int)hostAuxIn[i]);
  }
  printf("\n");


  // 3. Scan the aux array:
  scan<<<aux_grid, aux_block>>>(deviceAuxIn, deviceAuxOut, grid_size);

  _check(cudaMemcpy(hostAuxOut, deviceAuxOut, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < grid_size; i++) {
    printf("%d, ",(int)hostAuxOut[i]);
  }
  printf("\n");

  // 4. Add the aux to the individual scanned blocks:
  add<<<scan_grid, scan_block>>>(deviceOutput, deviceAuxOut, numElements);

  cudaDeviceSynchronize();

  _check(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));

  for(int i = 0; i < numElements; i++) {
    printf("%d, ", (int)hostOutput[i]);
  }
  printf("\n");

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  free(hostInput);
  free(hostOutput);

  return 0;
} 
