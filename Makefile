CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -Ofast -march=native
GCC_CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -Ofast -march=native
CLANG_CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -O3 -ffast-math -march=native
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=solver.o wtime.o

all: $(TARGETS)

all-gcc: CC=gcc
all-gcc: CFLAGS=$(GCC_CFLAGS)
all-gcc: $(TARGETS)

all-clang: CC=clang
all-clang: CFLAGS=$(CLANG_CFLAGS)
all-clang: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $(CC)-$@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $(CC)-$@ $(LDFLAGS)

clean:
	rm -f $(addprefix *, $(TARGETS)) *.o *~ .depend

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
