# Target rules

CC = g++

OPT = -O3 -DMKL_ILP64 -m64 -I"${MKLROOT}/include"
LIBRARIES = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

################################################################################

# Target rules

all: build
build: vd

VD.o: VD.cpp
	$(CC) $(OPT) -o $@ -c $< -w

vd: VD.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES)

clean:
	-rm -f vd VD.o RS_algorithm.o

.PHONY: clean run all
