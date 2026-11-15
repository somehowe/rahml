CXX := g++-15
CXXFLAGS := -std=c++20 -fmodules-ts -Wall -Wextra -Wpedantic -g

.PHONY: all run clean

all: build/demo

build:
	mkdir -p build

build/headers.stamp: Makefile | build
	$(CXX) $(CXXFLAGS) -x c++-system-header -c iostream memory
	touch build/headers.stamp

build/rahml.o: src/rahml.cc build/headers.stamp
	$(CXX) $(CXXFLAGS) -c src/rahml.cc -o build/rahml.o

build/demo: examples/demo.cc build/rahml.o
	$(CXX) $(CXXFLAGS) examples/demo.cc build/rahml.o -o build/demo

run: build/demo
	./build/demo

clean:
	rm -rf build gcm.cache
