  # Target rules

LIBRARIES := -lblas -lcublas -lcusolver -lm

CC = nvc

OPT := -fast -mp=gpu -Minfo=accel -Mcudalib=cublas,cusolver -DDEBUG

################################################################################

# Target rules

all: build
build: vca

VCA.o: VCA.cpp
	$(CC) $(OPT) -o $@ -c $< -w

vca: VCA.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES) -w

clean:
	rm -f VCA.o vca
