all : main.o
	nvcc -g -G -L../WebGPU/src/wb/c-tools/Linux-x86_64/ -lwb main.o -o histogram

main.o : main.cu
	nvcc -g -G -I../WebGPU/src/wb/c-tools/ main.cu -c -o main.o

clean:
	rm *.o histogram
