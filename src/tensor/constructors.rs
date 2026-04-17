use crate::grad::BackpropOp;
use crate::rawtensor::RawTensor;
use crate::tensor::core::TensorInner;
use super::Tensor;
use std::rc::Rc;
use std::cell::RefCell;


impl Tensor {
    pub(crate) fn from_inner(tensor: TensorInner) -> Tensor {
        Tensor(Rc::new(tensor))
    }

    pub(crate) fn from_raw(raw: RawTensor, grad: Option<RawTensor>) -> Tensor {
        Tensor::from_inner (
            TensorInner {
                raw,
                grad: RefCell::new(grad),
                inputs: Box::from([]),
                op: BackpropOp::None,
                requires_grad: false,
            }
        )
    }


    // Creates a RawTensor from by copying data from a slice
    // Panics if shape and data don't match lengths
    pub fn from_slice(shape: &[usize], data: &[f64]) -> Tensor {
        Tensor::from_raw(RawTensor::from_slice(shape, data), None)
    }

    // Creates a RawTensor from by using data from a Vec (no copying done here)
    // Panics if shape and data don't match lengths
    // The Vec loses its ownership of the data
    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> Tensor {
        Tensor::from_raw(RawTensor::from_vec(shape, data), None)
    }

    // Creates a Tensor from by using data from a Box (no copying done here)
    // Panics if shape and data don't match lengths
    // The Box loses its ownership of the data
    pub fn from_box(shape: &[usize], data: Box<[f64]>) -> Tensor {
        Tensor::from_raw(RawTensor::from_box(shape, data), None)
    }

    pub fn full(shape: &[usize], value: f64) -> Tensor {
        Tensor::from_raw(RawTensor::full(shape, value), None)
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn linspace(start: f64, end: f64, n: usize) -> Tensor {
        Tensor::from_raw(RawTensor::linspace(start, end, n), None)
    }

    pub fn rand_range(shape: &[usize], l: f64, r: f64) -> Tensor {
        Tensor::from_raw(RawTensor::rand_range(shape, l, r), None)
    }

    pub fn rand(shape: &[usize]) -> Tensor {
        Tensor::rand_range(shape, 0.0, 1.0)
    }

    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> Tensor {
        Tensor::from_raw(RawTensor::randn(shape, mean, std_dev), None)
    }
}

