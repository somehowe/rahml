CXX := g++-15.2.0
CXXFLAGS := -std=c++20 -fmodules-ts -Wall -Wextra -Wpedantic -g

RAHML_OBJS := build/rahml.o build/autograd.o

.PHONY: all run clean test regression

all: build/demo

build:
	mkdir -p build

build/headers.stamp: Makefile | build
	$(CXX) $(CXXFLAGS) -x c++-system-header -c iostream memory vector \
		utility cassert unordered_set algorithm string cmath stdexcept
	touch build/headers.stamp

build/rahml.o: src/rahml.cc build/headers.stamp
	$(CXX) $(CXXFLAGS) -c src/rahml.cc -o build/rahml.o

build/autograd.o: src/autograd.cc build/rahml.o
	$(CXX) $(CXXFLAGS) -c src/autograd.cc -o build/autograd.o

build/demo: examples/demo.cc $(RAHML_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/test: tests/value_test.cc $(RAHML_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

build/linear_regression: examples/linear_regression.cc $(RAHML_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

regression: build/linear_regression
	./build/linear_regression

run: build/demo
	./build/demo 2 3

test: build/test
	./build/test

clean:
	rm -rf build gcm.cache
