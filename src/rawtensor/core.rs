use std::rc::Rc;

#[derive(Clone)] 
pub struct RawTensor {
    pub(super) shape: Box<[usize]>,
    pub(super) strides: Box<[usize]>,
    pub(super) data: Rc<[f64]>,
}

impl RawTensor {
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn data(&self) -> &[f64]    { &self.data  }
}

impl PartialEq for RawTensor {
    fn eq(&self, other: &RawTensor) -> bool {
        self.shape == other.shape && 
        self.get_contiguous() == other.get_contiguous()
    }
}
