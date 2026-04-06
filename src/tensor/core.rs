use crate::rawtensor::RawTensor;
use core::fmt;
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};
use crate::grad::BackpropOp;

pub struct TensorInner {
    pub raw: RawTensor,
    pub grad: Option<RawTensor>,
    pub inputs: Box<[Tensor]>,
    pub backprop_op: BackpropOp,
    pub requires_grad: bool,
}

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInner>>);

impl Tensor {
    pub fn borrow(&self) -> Ref<TensorInner> {
        Ref::map(self.0.borrow(), |t| t)
    }

    pub fn borrow_mut(&self) -> RefMut<TensorInner> {
        RefMut::map(self.0.borrow_mut(), |t| t)
    }

    pub fn shape(&self) -> Box<[usize]> {
        Box::from(self.0.borrow().raw.shape())
    }

    pub fn data(&self) -> Box<[f64]> {
        Box::from(self.0.borrow().raw.data())
    }

    pub fn set_requires_grad(&self, requires_grad: bool) {
        self.borrow_mut().requires_grad = requires_grad;
    }

    pub fn get_requires_grad(&self) -> bool {
        self.borrow().requires_grad
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        self.borrow().raw == other.borrow().raw
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor {{ data: {:?}", self.borrow().raw)?;

 
        if let Some(grad) = self.borrow().grad.as_ref() {
            write!(f, ", grad: {}", grad)?;
        }

        write!(f, " }}")
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor {{ data: {:?}", self.borrow().raw)?;

        if let Some(grad) = self.borrow().grad.as_ref() {
            write!(f, ", grad: {}", grad)?;
        }

        write!(f, " }}")
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

