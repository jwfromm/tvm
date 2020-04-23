CC=g++

TVM_ROOT=../../..

INCLUDES=-I${TVM_ROOT}/include
INCLUDES+=-I${TVM_ROOT}/3rdparty/dmlc-core/include
INCLUDES+=-I${TVM_ROOT}/3rdparty/dlpack/include

CFLAGS=-std=c++14 -fPIC -O3 ${INCLUDES}
LDFLAGS=-shared -L${TVM_ROOT}/build -ltvm

libhf_passes.so : hf_passes.cc
	$(CC) ${CFLAGS} $+ -o $@ ${LDFLAGS}

clean:
	rm -rf *.so