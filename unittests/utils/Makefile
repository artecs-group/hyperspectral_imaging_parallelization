# Default compiler and flags
CC = icpx
CFLAGS = -I"${MKLROOT}/include" -qmkl -DMKL_ILP64 -O3
CLINK = -lgtest -lgtest_main -lpthread

matrix_operations.exe: matrix_operations.cpp ../../src/utils/matrix_operations.hpp
	$(CC) $(CFLAGS) matrix_operations.cpp -o matrix_operations.exe $(CLINK)

run: matrix_operations.exe
	./matrix_operations.exe

.PHONY: clean
clean:
	rm -f matrix_operations.exe