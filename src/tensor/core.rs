use crate::rawtensor::RawTensor;
use core::fmt;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Tensor {
    pub(super) raw: Rc<RefCell<RawTensor>>,
    pub(super) requires_grad: bool,
    pub(super) inputs: Rc<[Tensor]>,
    pub(super) grad: Option<Rc<RefCell<RawTensor>>>,
    pub(super) backprop: Option<Rc<dyn Fn(&RawTensor)>>,
}

impl Tensor {
    pub fn shape(&self) -> Box<[usize]> {
        Box::from(self.raw.borrow().shape())
    }

    pub fn data(&self) -> Box<[f64]> {
        Box::from(self.raw.borrow().data())
    }
}

impl Clone for Tensor {                                                       
      fn clone(&self) -> Tensor {
          Tensor {                                                             
              raw: Rc::clone(&self.raw),
              requires_grad: self.requires_grad,
              grad: self.grad.as_ref().map(Rc::clone),
              inputs: Rc::clone(&self.inputs),
              backprop: self.backprop.as_ref().map(Rc::clone),
          }                                                                     
      }
  }                                                                             
   
impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        *self.raw.borrow() == *other.raw.borrow()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor {{ data: {:?}", self.raw.borrow())?;

        if self.grad.is_some() {
            write!(f, ", grad: <some>")?;
        }

        write!(f, " }}")
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor {{ data: {}", self.raw.borrow())?;

        if self.grad.is_some() {
            write!(f, ", grad: <some>")?;
        }

        write!(f, " }}")
    }
}


#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

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

