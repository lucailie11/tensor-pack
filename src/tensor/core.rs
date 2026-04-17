use core::fmt;
use std::ops::Deref;
use std::rc::Rc;
use std::cell::{RefCell};
use crate::rawtensor::RawTensor;
use crate::grad::BackpropOp;

pub struct TensorInner {
    pub raw: RawTensor,
    pub grad: RefCell<Option<RawTensor>>,
    pub inputs: Box<[Tensor]>,
    pub op: BackpropOp,
    pub requires_grad: bool,
}

#[derive(Clone)]
pub struct Tensor(pub(crate) Rc<TensorInner>);

impl Deref for Tensor {
    type Target = TensorInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        self.raw == other.raw
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor {{ data: {:?}", self.raw)?;

 
        if let Some(grad) = self.grad.borrow().as_ref() {
            write!(f, ", grad: {}", grad)?;
        }

        write!(f, " }}")
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor {{ data: {:?}", self.raw)?;

        if let Some(grad) = self.grad.borrow().as_ref() {
            write!(f, ", grad: {}", grad)?;
        }

        write!(f, " }}")
    }
}

impl Tensor {
    pub fn shape(&self) -> &[usize] { self.raw.shape() }
    pub fn data(&self) -> &[f64] { self.raw.data() }

    pub fn reshape(&mut self, new_shape: &[usize]) {
        if self.op == BackpropOp::None && let Some(tensor) = Rc::get_mut(&mut self.0) {
            tensor.raw.reshape(new_shape);
        } else {
            panic!("Can't reshape non-leaf tensor");
        }
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if let Some(tensor) = Rc::get_mut(&mut self.0) {
            tensor.requires_grad = requires_grad;
        } else {
            panic!("Can't change requires_grad on a tensor with at least one out edge");
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_eq() {
        let a: Tensor = Tensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut b: Tensor = Tensor::linspace(1.0, 6.0, 6);
        b.reshape(&[2, 3]);
        assert_eq!(a, b)
    }

    #[test]
    fn test_partial_eq_diff_shape() {
        let a: Tensor = Tensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b: Tensor = Tensor::linspace(1.0, 6.0, 6);
        assert_ne!(a, b)
    }

    #[test]
    fn test_partial_eq_diff_data() {
        let a: Tensor = Tensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut b: Tensor = Tensor::linspace(1.0, 7.0, 6);
        b.reshape(&[2, 3]);
        assert_ne!(a, b)
    }
}

