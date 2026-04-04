# Rust Tensor Library for ML

A tensor library written in Rust from scratch, inspired by NumPy and PyTorch. Built as a learning project with a focus on execution speed and clean architecture.

Tensors store data in a flat array using row-major order. Arithmetic operators are overloaded so you can write `a + b`, `a * 2.0`, etc. naturally.

## Architecture

The library is split into two layers:

- **`RawTensor`** — pure math. Handles all tensor operations (arithmetic, reductions, transformations). No gradient awareness.
- **`Tensor`** — the public-facing type. Wraps `RawTensor` in `Rc<RefCell<>>` for shared ownership, and carries `GradInfo` for automatic differentiation (in progress).

```
src/
  rawtensor/   — RawTensor: core math layer
  tensor/      — Tensor: grad-aware wrapper around RawTensor
  utils/       — shared helpers (strided slice operations)
```

## To do
- add more tests
- implement backpropagation
- optimizing broadcasting index lookup
- implement dot and cross product
- add new transformations
- concurrency maybe
- add to pyhton

## Features

### Constructors
- `zeros` / `ones` / `full`
- `linspace` — evenly spaced 1D tensor
- `rand` — sample from U([0, 1))
- `rand_range` — sample from U([l, r))
- `randn` — sample from N(mean, std_dev)
- `new` — construct from existing shape and data slices
- `reshape` — change shape without moving data

### Binary ops
Element-wise operations between two tensors with NumPy-style broadcasting.
`elementwise_op` / `elementwise_op_inplace` are the core primitives.
In-place variants require `self` to already hold the output shape.
- `&Tensor + &Tensor` / `Tensor += &Tensor`
- `&Tensor - &Tensor` / `Tensor -= &Tensor`
- `&Tensor * &Tensor` / `Tensor *= &Tensor`
- `&Tensor / &Tensor` / `Tensor /= &Tensor`

### Scalar ops
Arithmetic between a tensor and a scalar `f64`.
- `Tensor + f64` / `f64 + Tensor` / `Tensor += f64`
- `Tensor - f64` / `f64 - Tensor` / `Tensor -= f64`
- `Tensor * f64` / `f64 * Tensor` / `Tensor *= f64`
- `Tensor / f64` / `f64 / Tensor` / `Tensor /= f64`

### Matrix multiplication
- `a.matmul(&b)` — 2D only, shapes `[m,k] × [k,n]` → `[m,n]`

### Unary ops
Element-wise single-tensor operations. `map` / `map_inplace` are the core primitives.
- `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu` (and `_inplace` variants)
- `-Tensor` — negation

### Reductions
Reduce one axis, dropping it from the output shape. `reduce_axis` is the core primitive.
- `sum_axis`, `mean_axis`, `var_axis`, `std_dev_axis`

### Transformations
Axis-based operations applied independently along one axis. `apply_axis` / `apply_axis_inplace` are the core primitives.
- `softmax(axis)` / `softmax_inplace(axis)` — numerically stable (subtracts max before exp)
