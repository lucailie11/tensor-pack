# Rust Tensor Library for ML

A tensor library written in Rust from scratch. Tensors store data in a flat array using row-major order. All arithmetic operators are overloaded so you can write `&a + &b`, `&a * 2.0`, etc. naturally.
Inspired by NumPy and PyTorch, built in Rust as a learning project with a focus on execution speed.

## To do
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
Element-wise operations between two tensors. Shapes must match (broadcasting not yet supported).
- [x] `&Tensor + &Tensor` — element-wise addition
- [x] `Tensor += &Tensor`
- [x] `&Tensor - &Tensor` — element-wise subtraction
- [x] `Tensor -= &Tensor`

### Scalar ops (`scalar.rs`)
Arithmetic between a tensor and a scalar `f64`.
- [x] `&Tensor + f64` / `f64 + &Tensor`
- [x] `Tensor += f64`
- [x] `&Tensor - f64` / `f64 - &Tensor`
- [x] `Tensor -= f64`
- [x] `&Tensor * f64` / `f64 * &Tensor` — scalar scaling
- [x] `Tensor *= f64`
- [x] `&Tensor / f64` / `f64 / &Tensor`
- [x] `Tensor /= f64`

### Matrix multiplication (`matmul.rs`)
- [x] `&Tensor * &Tensor` — matrix multiplication (2D only, shapes `[m,k] × [k,n]`)
- [x] `Tensor *= &Tensor`

### Pointwise ops (`pointwise.rs`)
Element-wise unary operations. `map` / `map_inplace` are the core primitives.
- [x] `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu`
- [x] `*_inplace` variants for all of the above
- [x] `-&Tensor` — negation

### Reduction ops (`reduction.rs`)
Uses Welford's algorithm for numerically stable variance.
- [x] `sum` / `sum_axis(axis)`
- [x] `mean` / `mean_axis(axis)`
- [x] `var` / `var_axis(axis)` — population variance (σ²)
- [x] `std_dev` / `std_dev_axis(axis)` — population standard deviation (σ)

### Axis ops (`axis_ops.rs`)
Operations applied independently along one axis. `apply_axis` / `apply_axis_inplace` are the core
primitives; future normalizations (layer norm, batch norm) will use the same pattern.
- [x] `softmax(axis)` / `softmax_inplace(axis)` — stable softmax (subtracts max before exp)
