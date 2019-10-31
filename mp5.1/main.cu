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

int main(int argc, char** argv) {
  if(argc != 2) {
    printf("Usage: ./scan <input_data_file>\n");
    return -1;
  }

  return 0;
} 
