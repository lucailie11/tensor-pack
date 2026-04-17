# rml — Rust ML Tensor Library

A tensor library written in Rust from scratch, inspired by NumPy and PyTorch. Built as a learning project with a focus on clean architecture and automatic differentiation.

Tensors store data in a flat array in row-major order. Arithmetic operators are overloaded so you can write `&a + &b`, `&a * 2.0`, etc. naturally.

## Architecture

The library has four layers:

- **`rawtensor/`** — Handles all tensor operations (arithmetic, broadcasting, reductions, transformations). No gradient awareness.
- **`tensor/`** — Wraps `RawTensor` and carries the computation graph for automatic differentiation.
- **`grad/`** — autograd engine. Builds the computation graph during forward passes and runs backpropagation via topological sort.
- **`utils/`** — shared helpers (strided slice operations for reductions).

```
src/
  rawtensor/   — RawTensor: core math layer (no grad)
  tensor/      — Tensor: grad-aware public type
  grad/        — autograd engine (backward pass)
  utils/       — strided slice helpers
```

The only public type is `Tensor`. `RawTensor` and the grad internals are crate-private.

## Features

### Constructors
- `Tensor::zeros(shape)` / `ones(shape)` / `full(shape, value)`
- `Tensor::linspace(start, end, n)` — evenly spaced 1D tensor
- `Tensor::from_slice(shape, data)` / `from_vec` / `from_box`
- `Tensor::rand(shape)` — sample from U([0, 1))
- `Tensor::rand_range(shape, l, r)` — sample from U([l, r))
- `Tensor::randn(shape, mean, std_dev)` — sample from N(mean, std_dev)
- `tensor.reshape(new_shape)` — change shape in place

### Binary ops
Element-wise operations with NumPy-style broadcasting.
- `&Tensor + &Tensor`, `&Tensor - &Tensor`, `&Tensor * &Tensor`, `&Tensor / &Tensor`

### Scalar ops
Arithmetic between a tensor and an `f64`, both orderings supported.
- `&Tensor + f64` / `f64 + &Tensor`
- `&Tensor - f64` / `f64 - &Tensor`
- `&Tensor * f64` / `f64 * &Tensor`
- `&Tensor / f64` / `f64 / &Tensor`

### Unary ops
- `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu`
- `-&Tensor` — negation

### Matrix multiplication
- `tensor.matmul(other)` — 2D only, shapes `[m, k] × [k, n]` → `[m, n]`

### Reductions
Reduce along one axis, dropping it from the output shape.
- `sum_axis(axis)`, `mean_axis(axis)`, `var_axis(axis)`, `std_dev_axis(axis)`

### Transformations
- `softmax(axis)` / `softmax_inplace(axis)` — numerically stable

### Autograd
Gradients are tracked automatically during the forward pass. Call `.backward()` on any tensor to populate `.grad` on all leaf tensors that have `requires_grad = true`.

Supported ops for backprop: `+`, `-`, `*`, `/` (tensor-tensor and scalar variants), `exp`, `ln`, `sqrt`.

> **Not yet implemented:** `sigmoid`, `tanh`, `abs`, `relu` backprop, and gradient support for `matmul`, reductions, and transformations.

## Usage

```rust
use rml::Tensor;

// construct tensors
let mut x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
let mut y = Tensor::randn(&[3], 0.0, 1.0);

// enable gradient tracking on leaf tensors
x.set_requires_grad(true);
y.set_requires_grad(true);

// build a computation graph with normal operators
let z = &x * &y;          // element-wise multiply
let loss = &z.exp() + 1.0; // exp then add scalar

// run backprop — populates x.grad and y.grad
loss.backward();

println!("{:?}", x); // Tensor { data: ..., grad: ... }
println!("{:?}", y);
```

## To do
- Implement backprop for `sigmoid`, `tanh`, `abs`, `relu`
- Gradient support for `matmul`, reductions, and `softmax`
- Optimise broadcasting index lookup
- Concurrency support
- Python support
