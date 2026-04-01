# Rust Tensor Library for ML

A tensor library written in Rust from scratch. Tensors store data in a flat array using row-major order. All arithmetic operators are overloaded so you can write `&a + &b`, `&a * 2.0`, etc. naturally.
Inspired by NumPy and PyTorch, built in Rust as a learning project with a focus on execution speed.

## To do
- implement full reductions (sum, mean, variance, std-dev over the whole tensor)
- gradients

## Features

### Core (`core.rs`)
- [x] Tensor struct (`shape: Box<[usize]>`, `data: Box<[f64]>`)
- [x] `zeros` / `ones` / `full` constructors
- [x] `linspace` — evenly spaced 1D tensor
- [x] `new` — construct from a data slice
- [x] `randn` — sample from N(mean, variance)
- [x] `reshape` — change shape without moving data

### Addition (`add.rs`)
- [x] `&Tensor + &Tensor` — element-wise
- [x] `Tensor += &Tensor`
- [x] `&Tensor + f64` / `f64 + &Tensor` — scalar broadcast
- [x] `Tensor += f64`

### Subtraction (`sub.rs`)
- [x] `&Tensor - &Tensor` — element-wise
- [x] `Tensor -= &Tensor`
- [x] `&Tensor - f64` / `f64 - &Tensor` — scalar broadcast
- [x] `Tensor -= f64`

### Multiplication (`mul.rs`)
- [x] `&Tensor * &Tensor` — matrix multiplication (2D only, shapes [m,k] × [k,n])
- [x] `Tensor *= &Tensor`
- [x] `&Tensor * f64` / `f64 * &Tensor` — scalar scaling
- [x] `Tensor *= f64`

### Division (`div.rs`)
- [x] `&Tensor / f64` — scalar division
- [x] `Tensor /= f64`

### Unary ops (`unary.rs`)
- [x] `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu`
- [x] `*_inplace` variants for all of the above
- [x] `-&Tensor` — negation

### Reduction ops (`red.rs`)
- [x] `sum_axis(axis)` — sum along one axis
- [ ] `sum` — sum of all elements
- [ ] `mean` / `mean_axis(axis)`
- [ ] `variance` / `variance_axis(axis)`
- [ ] `std_dev` / `std_axis(axis)`

### Activation functions
- [x] `relu`, `sigmoid`, `tanh` (covered under unary ops)
- [ ] `softmax`
