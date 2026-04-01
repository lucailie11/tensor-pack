use crate::Tensor;
use std::ops::Neg;

// Element-wise unary operations. Each function has an `_inplace` variant that
// mutates the tensor in place instead of allocating a new one.
// Supported: exp, ln, sqrt, abs, tanh, sigmoid, relu.
//
// The unary negation operator (-&Tensor) is also implemented here.

fn sigmoid(x: f64) -> f64 {
    1.0 / ((-x).exp() + 1.0)
}

fn relu(x: f64) -> f64 {
    if x >= 0.0 { x } else { 0.0 }
}

impl Tensor {
    fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .map(|&x| f(x))
                .collect::<Vec<f64>>()
                .into_boxed_slice(),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.data.iter_mut().for_each(|x| *x = f(*x));
    }

    pub fn exp(&self) -> Tensor {
        self.map(f64::exp)
    }
    pub fn ln(&self) -> Tensor {
        self.map(f64::ln)
    }
    pub fn sqrt(&self) -> Tensor {
        self.map(f64::sqrt)
    }
    pub fn abs(&self) -> Tensor {
        self.map(f64::abs)
    }
    pub fn tanh(&self) -> Tensor {
        self.map(f64::tanh)
    }
    pub fn sigmoid(&self) -> Tensor {
        self.map(sigmoid)
    }
    pub fn relu(&self) -> Tensor {
        self.map(relu)
    }

    pub fn exp_inplace(&mut self) {
        self.map_inplace(f64::exp)
    }
    pub fn ln_inplace(&mut self) {
        self.map_inplace(f64::ln)
    }
    pub fn sqrt_inplace(&mut self) {
        self.map_inplace(f64::sqrt)
    }
    pub fn abs_inplace(&mut self) {
        self.map_inplace(f64::abs)
    }
    pub fn tanh_inplace(&mut self) {
        self.map_inplace(f64::tanh)
    }
    pub fn sigmoid_inplace(&mut self) {
        self.map_inplace(sigmoid)
    }
    pub fn relu_inplace(&mut self) {
        self.map_inplace(relu)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self * (-1.0)
    }
}
