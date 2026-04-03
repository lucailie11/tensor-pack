# Rust Tensor Library for ML

A tensor library written in Rust from scratch. Tensors store data in a flat array using row-major order. All arithmetic operators are overloaded so you can write `&a + &b`, `&a * 2.0`, etc. naturally.
Inspired by NumPy and PyTorch, built in Rust as a learning project with a focus on execution speed.

## To do
- broadcasting
- gradients

## Features

### Core (`core.rs`)
- [x] Tensor struct (`shape: Box<[usize]>`, `data: Box<[f64]>`)
- [x] `zeros` / `ones` / `full` constructors
- [x] `linspace` — evenly spaced 1D tensor
- [x] `new` — construct from a data slice
- [x] `randn` — sample from N(mean, variance)
- [x] `reshape` — change shape without moving data

### Binary ops (`binary.rs`)
Element-wise operations between two tensors. `elementwise_op` / `elementwise_op_inplace` are the core primitives.
Shapes must match exactly (broadcasting not yet supported).
- [x] `&Tensor + &Tensor` / `Tensor += &Tensor`
- [x] `&Tensor - &Tensor` / `Tensor -= &Tensor`
- [x] `&Tensor * &Tensor` / `Tensor *= &Tensor` — element-wise multiplication
- [x] `&Tensor / &Tensor` / `Tensor /= &Tensor` — element-wise division

### Scalar ops (`scalar.rs`)
Arithmetic between a tensor and a scalar `f64`. All ops delegate to `map` / `map_inplace`.
- [x] `&Tensor + f64` / `f64 + &Tensor` / `Tensor += f64`
- [x] `&Tensor - f64` / `f64 - &Tensor` / `Tensor -= f64`
- [x] `&Tensor * f64` / `f64 * &Tensor` / `Tensor *= f64`
- [x] `&Tensor / f64` / `f64 / &Tensor` / `Tensor /= f64`

### Matrix multiplication (`matmul.rs`)
- [x] `a.matmul(&b)` — matrix multiplication (2D only, shapes `[m,k] × [k,n]` → `[m,n]`)

### Unary ops (`unary.rs`)
Element-wise single-tensor operations. `map` / `map_inplace` are the core primitives.
- [x] `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu` (and `*_inplace` variants)
- [x] `-&Tensor` — negation

### Reduction ops (`reduction.rs`)
Uses Welford's algorithm for numerically stable single-pass variance.
- [x] `sum_axis(axis)` / `mean_axis(axis)` / `var_axis(axis)` / `std_dev_axis(axis)`

### Transformations (`transformations.rs`)
Axis-based operations applied independently along one axis. `apply_axis` / `apply_axis_inplace`
are the core primitives; future normalizations (layer norm, batch norm) will follow the same pattern.
- [x] `softmax(axis)` / `softmax_inplace(axis)` — numerically stable (subtracts max before exp)
