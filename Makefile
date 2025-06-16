N?=512
# Compiler flags
NVCC = nvcc
NVCCFLAGS = -O3 -D N_VALUE=$(N)

# Source files
SRC = headless.c wtime.c solver.cu demo.c

# Object files
COMMON_OBJ = solver.o wtime.o

all: $(TARGET)

demo.o: demo.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

headless.o: headless.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

wtime.o: wtime.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

solver.o: solver.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link everything
headless: $(COMMON_OBJ) headless.o
	$(NVCC) $(NVCCFLAGS) headless.o $(COMMON_OBJ) -o headless

demo: $(COMMON_OBJ) demo.o
	$(NVCC) $(NVCCFLAGS) -lGL -lGLU -lglut demo.o $(COMMON_OBJ) -o demo

# Clean rule
clean:
	rm -f *.o $(TARGET)

