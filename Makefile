all: test kernel.cubin

kernel.cubin: kernel.cu
	nvcc kernel.cu -cubin -arch sm_10 -Xptxas -v -o kernel.cubin

test: test.cpp cuda_helper.h cuda_context.h gpu_allocator.h
	c++ -Wall -std=c++11 test.cpp -o test -lcuda
