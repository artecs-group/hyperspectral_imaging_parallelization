# Target rules

LIBRARIES := -lblas -lcublas -lcusolver

OPT := -g -acc -Minfo=accel -Mcudalib=cublas,cusolver -DDEBUG -O4

CC = nvc

################################################################################

# Target rules

all: build
build: vd

VD.o: VD.cpp
	$(CC) $(OPT) -o $@ -c $< -w

vd: VD.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f VD.o vd
