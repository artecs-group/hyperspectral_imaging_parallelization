# Default compiler and flags
CC = icpx
CFLAGS = -I"${MKLROOT}/include" -qmkl -DMKL_ILP64 -DSEQUENTIAL -O3 -DDEBUG
CLINK = -lgtest -lgtest_main -lpthread -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

vca.exe: vca.cpp ../../src/VCA/sequential/vca.o ../../src/utils/file_utils.hpp
	$(CC) $(CFLAGS) vca.cpp ../../src/VCA/sequential/vca.o -o vca.exe $(CLINK)

../../src/VCA/sequential/vca.o:
	$(CC) $(CFLAGS) ../../src/VCA/sequential/vca.cpp -c -o ../../src/VCA/sequential/vca.o

run: vca.exe
	./vca.exe

.PHONY: clean
clean:
	rm -f vca.exe *.o ../../src/VCA/sequential/vca.o