use crate::grad::BackpropOp;
use crate::rawtensor::RawTensor;
use super::core::TensorInner;
use super::Tensor;
use std::rc::Rc;
use std::cell::RefCell;


impl Tensor {
    pub fn from_inner(tensor: TensorInner) -> Tensor {
        Tensor(Rc::new(RefCell::new(tensor)))
    }

    pub fn from_raw(raw: RawTensor, grad: Option<RawTensor>) -> Tensor {
        Tensor::from_inner(
            TensorInner {
                raw,
                grad,
                inputs: Box::from([]),
                backprop_op: BackpropOp::None,
                requires_grad: false,
            }
        )
    }

    pub fn from_slice(shape: &[usize], data: &[f64]) -> Tensor {
        Tensor::from_raw(RawTensor::from_slice(shape, data), None)
    }

    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> Tensor {
        Tensor::from_raw(RawTensor::from_vec(shape, data), None)
    }

    pub fn from_box(shape: &[usize], data: Box<[f64]>) -> Tensor {
        Tensor::from_raw(RawTensor::from_box(shape, data), None)
    }

    pub fn reshape(&mut self, new_shape: &[usize]) {
        self.borrow_mut().raw.reshape(new_shape);
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

