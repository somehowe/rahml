# rahml: a small neural network library in C++20
[![Build and demo](https://github.com/somehowe/rahml/actions/workflows/ci.yml/badge.svg)](https://github.com/somehowe/rahml/actions/workflows/ci.yml)

[What I am building](docs/proposal.md) & [Example Client](examples/scaled_sine_skip.cc)

rahml is a C++20 neural network lib in development. Currently, it supports scalar autodiff with addition, subtraction, multiplication, division, and unary negation.

## Linear regression demo

```bash
make regression
```

Trains `y = w*x + b` on examples from `y = 2*x + 1`.
The output shows decreasing loss and parameters approaching `w=2`
and `b=1`. Assertions check that training reaches those values.

Build requirements: GCC 15 (I am using 15.2.0) and Make

## Build and run
```bash
make
./build/demo 2 3
```

The demo calculates (x + y) * y and its derivatives:

```text
result = 15
x.grad = 3
y.grad = 8
```

## Tests
```bash
make test
```
