# Target rules
 
LIBRARIES := -fortranlibs -lblas -llapack -lm

CC = gcc

CFLAGS = -O3

PGCC = pgcc -Minfo=accel -fast -acc -ta=multicore -DDEBUG

################################################################################

# Target rules

all: build
build: vca

VCA.o: VCA.cpp
	$(PGCC) -o $@ -c $< -w

vca: VCA.o
	$(PGCC) $(CFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f VCA.o vca
