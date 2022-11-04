DEBUG = no

# Default compiler and flags
CC = icpx
CFLAGS = -I"${MKLROOT}/include" -qmkl
CLINK =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

# Debug Flags
ifeq ($(DEBUG),yes)
 CFLAGS += -g -O0
else
 CFLAGS += -O3
endif

main: ./VD/vd.o main.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./VD/vd.o -o main

main.o: ./VD/vd.o
	$(CC) $(CFLAGS) main.cpp -c -o main.o

./VD/vd.o: ./VD/vd.hpp
	$(CC) $(CFLAGS) ./VD/vd.cpp -c -o ./VD/vd.o

run: main
	./main ../data/Cuprite 5

.PHONY: clean
clean:
	rm -f main \
	*.o \
	./VD/*.o