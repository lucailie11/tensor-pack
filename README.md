# rml — Rust ML Tensor Library

A tensor library written in Rust from scratch, inspired by NumPy and PyTorch. Built as a learning project with a focus on clean architecture and execution speed.

Tensors store data in a flat array in row-major order. Arithmetic operators are overloaded so you can write `&a + &b`, `&a * 2.0`, `a += &b`, etc. naturally.

## Architecture

The library has four layers:

- **`rawtensor/`** — Core math layer: arithmetic, broadcasting, reductions, structural ops. No gradient awareness.
- **`tensor/`** — Wraps `RawTensor` and carries the computation graph for automatic differentiation.
- **`grad/`** — Autograd engine. Builds the computation graph during the forward pass and runs backpropagation via topological sort.

```
src/
  rawtensor/   — RawTensor: core math layer (no grad)
  tensor/      — Tensor: grad-aware public type
  grad/        — autograd engine (backward pass)
```

The only public type is `Tensor`. `RawTensor` and the grad internals are crate-private.

## Features

### Constructors
- `Tensor::zeros(shape)` / `ones(shape)` / `full(shape, value)`
- `Tensor::linspace(start, end, n)` — evenly spaced 1D tensor
- `Tensor::identity(n)` — n×n identity matrix
- `Tensor::from_slice(shape, data)` / `from_vec` / `from_box` / `from_rc`
- `Tensor::rand(shape)` — sample from U([0, 1))
- `Tensor::rand_range(shape, l, r)` — sample from U([l, r))
- `Tensor::randn(shape, mean, std_dev)` — sample from N(mean, std_dev)

### Structure ops
- `reshape(new_shape)`, `transpose(perm)`, `expand(new_shape)`
- `squeeze_axes(axes)` / `squeeze(axis)` / `squeeze_all()` / `unsqueeze(axis)`

### Binary ops
Elementwise operations with broadcasting. Assign ops are not in-place.
- `&Tensor + &Tensor`, `&Tensor - &Tensor`, `&Tensor * &Tensor`, `&Tensor / &Tensor`
- `Tensor += &Tensor`, `Tensor -= &Tensor`, `Tensor *= &Tensor`, `Tensor /= &Tensor`

### Scalar ops
Arithmetic between a tensor and an `f64`, both orderings supported. Assign ops are not in-place.
- `&Tensor + f64` / `f64 + &Tensor` and `+=` variants
- `&Tensor - f64` / `f64 - &Tensor` and `-=` variants
- `&Tensor * f64` / `f64 * &Tensor` and `*=` variants
- `&Tensor / f64` / `f64 / &Tensor` and `/=` variants

### Unary ops
- `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu`
- `-&Tensor` — negation

### Linear algebra
- `tensor.dot(other)` — 1D dot product
- `tensor.matmul(other)` — 2D only, shapes `[m, k] × [k, n]` → `[m, n]`

### Reductions
Reduce along one axis, dropping it from the output shape.
- `sum_axis(axis)`, `mean_axis(axis)`, `var_axis(axis)`, `std_dev_axis(axis)`

### Normalizations
- `softmax(axis)`

### Autograd
Gradients are tracked automatically during the forward pass. Call `.backward()` on any tensor to populate `.grad` on all leaf tensors that have `requires_grad = true`.

`.backward()` seeds the gradient with all-ones. It is intended for scalar outputs — calling it on a non-scalar is valid but equivalent to summing all output elements before backpropagating.

Supported ops for backprop: `+`, `-`, `*`, `/` (tensor-tensor and scalar variants), `exp`, `ln`, `sqrt`, `abs`, `relu`, `sigmoid`.

> **Not yet implemented:** `tanh`, backprop (forward works, gradient is wrong).

## Usage

```rust
use rml::Tensor;

// construct tensors
let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
let y = Tensor::randn(&[3], 0.0, 1.0).requires_grad();

// build a computation graph with normal operators
let z = &x * &y;           // element-wise multiply
let loss = &z.exp() + 1.0; // exp then add scalar

// run backprop — populates x.grad and y.grad
loss.backward();

println!("{}", x);
println!("{}", y);
```

## To do
- More module tests
- Gradient supports
- Concurrency support
- Python bindings
