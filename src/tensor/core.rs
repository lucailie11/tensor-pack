use crate::rawtensor::RawTensor;
use super::grad::GradInfo;
use core::fmt;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Tensor {
    pub(super) raw: Rc<RefCell<RawTensor>>,
    pub(super) grad_info: Option<Rc<RefCell<GradInfo>>>,
}

impl Tensor {
    pub fn shape(&self) -> Box<[usize]> {
        Box::from(self.raw.borrow().shape())
    }

    pub fn data(&self) -> Box<[f64]> {
        Box::from(self.raw.borrow().data())
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.raw.borrow())               
    }
}
