# Target rules

LIBRARIES := -lblas -lcublas -lcusolver

OPT := -g -acc -Minfo=accel -Mcudalib=cublas,cusolver -DDEBUG -O4

CC = nvc

################################################################################

# Target rules

all: build
build: vca

VCA.o: VCA.cpp
	$(CC) $(OPT) -o $@ -c $< -w

vca: VCA.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f VCA.o vca
