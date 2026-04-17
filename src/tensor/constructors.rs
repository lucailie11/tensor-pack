use super::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use crate::grad::BackpropOp;
use crate::rawtensor::RawTensor;
use crate::tensor::core::TensorInner;


impl Tensor {
    pub(crate) fn from_inner(tensor: TensorInner) -> Tensor {
        Tensor(Rc::new(tensor))
    }

    pub(crate) fn no_grad_tensor(raw: RawTensor) -> Tensor {
        Tensor::from_inner (
            TensorInner {
                raw,
                grad: RefCell::new(None),
                inputs: Box::from([]),
                op: BackpropOp::None,
                requires_grad: false,
            }
        )
    }

    // Creates a RawTensor from by copying data from a slice
    // Panics if shape and data don't match lengths
    pub fn from_slice(shape: &[usize], data: &[f64]) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::from_slice(shape, data))
    }

    // Creates a RawTensor from by using data from a Vec (no copying done here)
    // Panics if shape and data don't match lengths
    // The Vec loses its ownership of the data
    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::from_vec(shape, data))
    }

    // Creates a Tensor from by using data from a Box (no copying done here)
    // Panics if shape and data don't match lengths
    // The Box loses its ownership of the data
    pub fn from_box(shape: &[usize], data: Box<[f64]>) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::from_box(shape, data))
    }

    pub fn full(shape: &[usize], value: f64) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::full(shape, value))
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn linspace(start: f64, end: f64, n: usize) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::linspace(start, end, n))
    }

    pub fn rand_range(shape: &[usize], l: f64, r: f64) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::rand_range(shape, l, r))
    }

    pub fn rand(shape: &[usize]) -> Tensor {
        Tensor::rand_range(shape, 0.0, 1.0)
    }

    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> Tensor {
        Tensor::no_grad_tensor(RawTensor::randn(shape, mean, std_dev))
    }
}

