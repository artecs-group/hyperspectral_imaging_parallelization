# Makefile for GNU Make

default: run

all: run

run: vd

MKL_COPTS = -O2 -DMKL_ILP64 -qmkl=parallel -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

vd: VD.cpp
	dpcpp $^ -o $@ $(DPCPP_OPTS)

clean:
	-rm -f vd

.PHONY: clean run all
