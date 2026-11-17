CXX := g++-15
CXXFLAGS := -std=c++20 -fmodules-ts -Wall -Wextra -Wpedantic -g

.PHONY: all run clean test

all: build/demo

build:
	mkdir -p build

build/headers.stamp: Makefile | build
	$(CXX) $(CXXFLAGS) -x c++-system-header -c iostream memory vector \
		utility cassert unordered_set
	touch build/headers.stamp

build/rahml.o: src/rahml.cc build/headers.stamp
	$(CXX) $(CXXFLAGS) -c src/rahml.cc -o build/rahml.o

build/demo: examples/demo.cc build/rahml.o
	$(CXX) $(CXXFLAGS) examples/demo.cc build/rahml.o -o build/demo

build/test: build/rahml.o tests/value_test.cc
	$(CXX) $(CXXFLAGS) tests/value_test.cc build/rahml.o -o build/test

run: build/demo
	./build/demo

test: build/test
	./build/test

clean:
	rm -rf build gcm.cache
