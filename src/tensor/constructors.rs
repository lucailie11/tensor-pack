use super::Tensor;
use crate::grad::BackpropOp;
use crate::rawtensor::RawTensor;
use crate::tensor::core::TensorInner;

use std::rc::Rc;
use std::cell::RefCell;

impl Tensor {
    pub(crate) fn from_inner(tensor: TensorInner) -> Tensor {
        Tensor(Rc::new(tensor))
    }

    // Wraps a RawTensor in a Tensor with no gradient tracking
    pub(super) fn no_grad_tensor(raw: RawTensor) -> Tensor {
        Tensor::from_inner(
            TensorInner {
                raw,
                grad: RefCell::new(None),
                inputs: Box::from([]),
                op: BackpropOp::None,
                requires_grad: false,
            }
        )
    }

    // Returns a Tensor from a reference to an Rc (no copying)
    pub fn from_rc(shape: &[usize], data: Rc<[f64]>) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::from_rc(shape, data))
    }
    // Returns a Tensor from a Box (copies data)
    pub fn from_box(shape: &[usize], data: Box<[f64]>) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::from_box(shape, data))
    }

    // Returns a Tensor from a Vec (copies data)
    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::from_vec(shape, data))
    }

    // Returns a Tensor from a slice (copies data)
    pub fn from_slice(shape: &[usize], data: &[f64]) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::from_slice(shape, data))
    }
    
    // Returns a 0D Tensor from a scalar (f64)
    pub fn from_scalar(scalar: f64) -> Tensor {
        Tensor::from_slice(&[], &[scalar])
    }

    // Returns a new Tensor with data in logical order
    pub fn contiguous(&self) -> Tensor {
        Tensor::no_grad_tensor(self.raw.contiguous())
    }

    // Creates a Tensor filled with value
    pub fn full(shape: &[usize], value: f64) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::full(shape, value))
    }

    // Creates a Tensor filled with 0.0
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    // Creates a Tensor filled with 1.0
    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    // Creates a 1D Tensor of n evenly spaced values in [start, end] (inclusive).
    // If n = 1, returns a shape-[1] tensor containing just start.
    pub fn linspace(start: f64, end: f64, n: usize) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::linspace(start, end, n))
    }

    // Creates a Tensor equal to I_n (the identity matrix of size [n x n])
    pub fn identity(n: usize) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::identity(n))
    }

    // Seeds the global RNG used by all random constructors
    pub fn set_seed(seed: u64) {
        RawTensor::set_seed(seed);
    }

    // Creates a Tensor filled with random samples from U([l, r))
    pub fn rand_range(shape: &[usize], l: f64, r: f64) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::rand_range(shape, l, r))
    }

    // Creates a Tensor filled with random samples from U([0, 1))
    pub fn rand(shape: &[usize]) -> Tensor {
        Tensor::rand_range(shape, 0.0, 1.0)
    }

    // Creates a Tensor filled with random samples from N(mean, std_dev)
    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::randn(shape, mean, std_dev))
    }
}

