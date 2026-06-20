# rml — Rust ML Tensor Library

A tensor library written in Rust from scratch, inspired by NumPy and PyTorch. Built as a learning project with a focus on clean architecture and execution speed.

Tensors store data in a flat array in row-major order. Arithmetic operators are overloaded so you can write `&a + &b`, `&a * 2.0`, `a += &b`, etc. naturally.

## Architecture

The library has three layers:

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
- `squeeze(axis)`, `unsqueeze(axis)`

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

`var` and `std_dev` use population variance (divides by n, not n-1).

### Normalizations
- `softmax(axis)`

### Autograd
Gradients are tracked automatically during the forward pass. Call `.backward()` on any tensor to populate `.grad` on all leaf tensors that have `requires_grad = true`.

`.backward()` seeds the gradient with all-ones. It is intended for scalar outputs — calling it on a non-scalar is equivalent to summing all output elements before backpropagating.

**Graph consumption:** `.backward()` consumes the computation graph. After it returns, `inputs` and `op` are cleared on every intermediate node, and intermediate gradients are freed. Only leaf gradients (`.requires_grad = true`) are kept. To run another backward pass on the same leaves, call `.zero_grad()` on each leaf first and rebuild the graph by rerunning the forward pass.

```
x.zero_grad();
let loss = forward(&x);
loss.backward();
// x.grad now holds the fresh gradient
```

**Supported ops for backprop:**

| Category   | Ops |
|------------|-----|
| Binary     | `+`, `-`, `*`, `/` (tensor–tensor and scalar variants) |
| Unary      | `exp`, `ln`, `sqrt`, `abs`, `relu`, `sigmoid`, `tanh` |
| Reductions | `sum_axis`, `mean_axis` |
| Linalg     | `dot`, `matmul` |
| Structure  | `reshape`, `transpose`, `expand`, `squeeze`, `unsqueeze` |

`softmax`, `var_axis`, and `std_dev_axis` detach from the computation graph — gradients do not flow through them.

## Usage

```
use rml::Tensor;

// construct tensors
let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
let y = Tensor::randn(&[3], 0.0, 1.0).requires_grad();

// build a computation graph with normal operators
let z = (&x * &y).relu();
let loss = z.sum_axis(0);

// run backprop — populates x.grad and y.grad
loss.backward();

// reuse leaves: clear old grads, rerun forward
x.zero_grad();
y.zero_grad();
let loss2 = (&x * &y).relu().sum_axis(0);
loss2.backward();
```
