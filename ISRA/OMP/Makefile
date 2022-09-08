# Target rules

LIBRARIES := -lblas -lcublas

OPT := -fast -mp=gpu -Minfo=accel -Mcudalib=cublas -DDEBUG -O4

CC = nvc

################################################################################

# Target rules

all: build
build: isra

ISRA.o: ISRA.cpp
	$(CC) $(OPT) -o $@ -c $< -w

RS_algorithm.o: RS_algorithm.cpp
	$(CC) $(OPT) -o $@ -c $< -w

isra: ISRA.o RS_algorithm.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) -w

clean:
	rm -f ISRA.o RS_algorithm.o isra
