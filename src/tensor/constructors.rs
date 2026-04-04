use std::rc::Rc;
use std::cell::RefCell;
use crate::rawtensor::RawTensor;
use super::Tensor;
use crate::grad::GradInfo;

impl Tensor {
    pub(super) fn new_tensor(data: RawTensor, grad_info: Option<GradInfo>) -> Tensor {
        Tensor {
            raw: Rc::new(RefCell::new(data)),
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
        (*self.raw).borrow_mut().reshape(new_shape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_shape_and_data() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(&*t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ones_shape_and_data() {
        let t = Tensor::ones(&[2, 3]);
        assert!(t.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn full_fills_value() {
        let t = Tensor::full(&[3], 7.0);
        assert!(t.data().iter().all(|&x| x == 7.0));
    }

    #[test]
    fn linspace_endpoints() {
        let t = Tensor::linspace(0.0, 1.0, 5);
        let d = t.data();
        assert!((d[0] - 0.0).abs() < 1e-9);
        assert!((d[4] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn new_from_slice() {
        let t = Tensor::new(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&*t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reshape_changes_shape() {
        let mut t = Tensor::linspace(1.0, 6.0, 6);
        t.reshape(&[2, 3]);
        assert_eq!(&*t.shape(), &[2, 3]);
    }

    #[test]
    fn rand_in_range() {
        let t = Tensor::rand(&[100]);
        assert!(t.data().iter().all(|&x| (0.0..1.0).contains(&x))); 
    }
}

