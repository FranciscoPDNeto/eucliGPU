CC := g++
CFLAGS := -Wall -O3 -std=c++17 -lOpenCL

all: eucligpu

eucligpu: eucligpu.o
	$(CC) $(CFLAGS) -o $@ $^
	rm -rf eucligpu.o

eucligpu.o: eucligpu.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm gpuTest