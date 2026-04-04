use std::rc::Rc;
use std::cell::RefCell;
use crate::rawtensor::RawTensor;
use super::Tensor;
use super::grad::GradInfo;

impl Tensor {
    pub fn new_tensor(data: RawTensor, grad_info: Option<GradInfo>) -> Tensor {
        Tensor {
            data: Rc::new(RefCell::new(data)),
            grad_info: grad_info.map(|g| Rc::new(RefCell::new(g))),
        }
    }

    pub fn full(shape: &[usize], value: f64) -> Tensor {
        Tensor::new_tensor(RawTensor::full(shape, value), None)
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn linspace(start: f64, end: f64, n: usize) -> Tensor {
        Tensor::new_tensor(RawTensor::linspace(start, end, n), None)
    }

    pub fn rand_range(shape: &[usize], l: f64, r: f64) -> Tensor {
        Tensor::new_tensor(RawTensor::rand_range(shape, l, r), None)
    }

    pub fn rand(shape: &[usize]) -> Tensor {
        Tensor::rand_range(shape, 0.0, 1.0)
    }

    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> Tensor {
        Tensor::new_tensor(RawTensor::randn(shape, mean, std_dev), None)
    }

    pub fn new(shape: &[usize], data: &[f64]) -> Tensor {
        Tensor::new_tensor(RawTensor::new(shape, data), None)
    }

    pub fn reshape(&mut self, new_shape: &[usize]) {
        (*self.data).borrow_mut().reshape(new_shape);
    }
}

