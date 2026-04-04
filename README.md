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
- [x] `new` — construct by copying the data and shape from slices
- [x] `rand` — sample from U([0, 1))
- [x] `rand_range` — sample from U([l, r))
- [x] `randn` — sample from N(mean, std_dev)
- [x] `reshape` — change shape without moving data

### Binary ops (`binary.rs`)
Element-wise operations between two tensors with NumPy-style broadcasting.
`elementwise_op` / `elementwise_op_inplace` are the core primitives (and can be used on their own).
In-place variants require `self` to already hold the output shape.
- [x] `&Tensor + &Tensor` / `Tensor += &Tensor`
- [x] `&Tensor - &Tensor` / `Tensor -= &Tensor`
- [x] `&Tensor * &Tensor` / `Tensor *= &Tensor`
- [x] `&Tensor / &Tensor` / `Tensor /= &Tensor`

### Scalar ops (`scalar.rs`)
Arithmetic between a tensor and a scalar `f64`. All ops delegate to `map` / `map_inplace`.
- [x] `&Tensor + f64` / `f64 + &Tensor` / `Tensor += f64`
- [x] `&Tensor - f64` / `f64 - &Tensor` / `Tensor -= f64`
- [x] `&Tensor * f64` / `f64 * &Tensor` / `Tensor *= f64`
- [x] `&Tensor / f64` / `f64 / &Tensor` / `Tensor /= f64`

### Matrix multiplication (`matmul.rs`)
- [x] `a.matmul(&b)` — matrix multiplication (2D only, shapes `[m,k] × [k,n]` → `[m,n]`)

### Unary ops (`unary.rs`)
Element-wise single-tensor operations. `map` / `map_inplace` are the core primitives (and can be used on their own).
- [x] `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu` 
- [x] `-&Tensor` — negation

### Reduction ops (`reductions.rs`)
Reduction operations that drop one axis. `reduce_axis` is the core primitive (and can be used on its own).
- [x] `sum_axis` 
- [x] `mean_axis` 
- [x] `var_axis`
- [x] `std_dev_axis`

### Transformations (`transformations.rs`)
Axis-based operations applied independently along one axis. `apply_axis` / `apply_axis_inplace`
are the core primitives (and can be used on their own) 
Future normalizations (layer norm, batch norm) will follow the same pattern.
- [x] `softmax(axis)` / `softmax_inplace(axis)` — numerically stable (subtracts max before exp)
