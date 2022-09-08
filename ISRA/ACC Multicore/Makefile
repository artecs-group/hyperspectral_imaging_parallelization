# Target rules

LIBRARIES := -lblas 

CC = gcc

CFLAGS = -O4

PGCC = pgcc -Minfo=accel -fast -acc -ta=multicore -DDEBUG

################################################################################

# Target rules

all: build
build: isra

ISRA.o: ISRA.cpp
	$(PGCC) $(CFLAGS) -o $@ -c $< -w

RS_algorithm.o: RS_algorithm.cpp
	$(PGCC) $(CFLAGS) -o $@ -c $< -w

isra: ISRA.o RS_algorithm.o
	$(PGCC) $(CFLAGS) -o $@ $+ $(LIBRARIES) -w

clean:
	rm -f ISRA.o RS_algorithm.o isra
