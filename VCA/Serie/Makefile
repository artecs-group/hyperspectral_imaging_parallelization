# Target rules

CC = gcc

OPT = -O3 -DMKL_ILP64 -m64 -I"${MKLROOT}/include"
LIBRARIES = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

################################################################################

# Target rules

all: build
build: vca

VCA_IT.o: VCA_IT.cpp
	$(CC) $(OPT) -o $@ -c $< -w

vca: VCA_IT.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES)

clean:
	-rm -f vd VCA_IT.o vca

.PHONY: clean run all
