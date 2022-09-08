# Target rules

CC = gcc

OPT = -O3 -DMKL_ILP64 -m64 -I"${MKLROOT}/include"
LIBRARIES = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

################################################################################

# Target rules

all: build
build: isra

ISRA.o: isra.cpp
	$(CC) $(OPT) -o $@ -c $< -w

RS_algorithm.o: RS_algorithm.cpp
	$(CC) $(OPT) -o $@ -c $< -w

isra: ISRA.o RS_algorithm.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES)

clean:
	rm -f isra ISRA.o RS_algorithm.o
