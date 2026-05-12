use crate::rawtensor::RawTensor;
use crate::grad::BackpropOp;

use std::ops::Deref;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
pub struct TensorInner {
    pub(crate) raw: RawTensor,
    pub(crate) grad: RefCell<Option<RawTensor>>,
    pub(crate) inputs: Box<[Tensor]>,
    pub(crate) op: BackpropOp,
    pub(crate) requires_grad: bool,
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

impl Tensor {
    pub fn shape(&self)    -> &[usize] { self.raw.shape() }
    pub fn data(&self)     -> &[f64]   { self.raw.data() }
    pub fn ndim(&self)     -> usize    { self.raw.ndim() }
    pub fn len(&self)      -> usize    { self.raw.len() }
    pub fn is_empty(&self) -> bool     { self.raw.is_empty() }
}

