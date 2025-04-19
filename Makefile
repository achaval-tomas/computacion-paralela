CC=gcc
CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -Ofast -march=native
GCC_CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -Ofast -march=native
CLANG_CFLAGS=-std=c11 -Wall -Wextra -Wno-unused-parameter -O3 -ffast-math -march=native -Rpass=loop-vectorize -Rpass-missed=loop-vectorize
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

test: test.c
	clang -O1 -ffast-math  -march=znver3 -ftree-vectorize  -Rpass=.* -Rpass-missed=.* test.c -S

test-rb: test-rb.c
	clang -O1 -ffast-math  -march=znver3 -ftree-vectorize  -Rpass=.* -Rpass-missed=.* test-rb.c -S

clean:
	rm -f $(addprefix *, $(TARGETS)) *.o *~ .depend *.s

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

sgemm:
	gcc -o time_sgemm time_sgemm.c -lopenblas

dgemm:
	gcc -o time_dgemm time_dgemm.c -lopenblas

-include .depend

.PHONY: clean all
