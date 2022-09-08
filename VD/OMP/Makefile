  # Target rules

LIBRARIES := -lblas -lcublas -lcusolver -lm

CC = nvc

OPT := -fast -mp=gpu -Minfo=accel -Mcudalib=cublas,cusolver -DDEBUG -O4

################################################################################

# Target rules

all: build
build: vd

VD.o: VD.cpp
	$(CC) $(OPT) -o $@ -c $< -w

vd: VD.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) -w

clean:
	rm -f VD.o vd
