# Makefile for GNU Make

default: run

all: run

run: isra

MKL_COPTS = -O2 -DMKL_ILP64 -qmkl=parallel -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_core -lOpenCL -liomp5 -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) $(MKL_LIBS)

isra: isra.cpp RS_algorithm.cpp
	dpcpp $^ -o $@ $(DPCPP_OPTS)

clean:
	-rm -f isra

.PHONY: clean run all
