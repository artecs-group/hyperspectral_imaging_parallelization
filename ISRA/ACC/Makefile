# Target rules

LIBRARIES := -lblas -lcublas -lcusolver

OPT := -g -acc -Minfo=accel -Mcudalib=cublas,cusolver -DDEBUG -O4

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
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f ISRA.o RS_algorith.o isra
