CXX      ?= g++
NVCC     ?= nvcc
ARCH     ?= sm_86
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wno-unused-function -Wno-unknown-pragmas
NVCCFLAGS ?= -O3 -arch=$(ARCH) -std=c++14 --expt-relaxed-constexpr

.PHONY: all cpu gpu clean

all: cpu gpu

cpu: vanity

vanity: vanity.cpp secp.h addr.h
	$(CXX) $(CXXFLAGS) -o vanity vanity.cpp

gpu: vanity_gpu

vanity_gpu: vanity.cu secp.h addr.h
	$(NVCC) $(NVCCFLAGS) -o vanity_gpu vanity.cu

clean:
	rm -f vanity vanity_gpu *.o
