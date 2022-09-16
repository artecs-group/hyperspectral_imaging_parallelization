# Target rules

LIBRARIES := -fortranlibs -lblas -llapack -lm

CC = gcc

CFLAGS = -O3

PGCC = pgcc -Minfo=accel -fast -acc -ta=multicore -DDEBUG

################################################################################

# Target rules

all: build
build: vd

VD.o: VD.cpp
	$(PGCC) $(CFLAGS) -o $@ -c $< -w

vd: VD.o
	$(PGCC) $(CFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f VD.o vd
