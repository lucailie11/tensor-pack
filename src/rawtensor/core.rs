use std::rc::Rc;

// RawTensor is a multidimensional array of f64 stored flat in data
// the dimension of the array are non-zero
// data is shared via Rc and may not be in logical order
// strides[i] gives the step in data for a unit increment along dimension i

#[derive(Clone)]
pub struct RawTensor {
    pub(super) shape:   Box<[usize]>,
    pub(super) strides: Box<[usize]>,
    pub(super) data:    Rc<[f64]>,
}

impl RawTensor {
    pub fn shape(&self)    -> &[usize] { &self.shape }
    pub fn data(&self)     -> &[f64]   { &self.data }
    pub fn ndim(&self)     -> usize    { self.shape.len() }
    pub fn len(&self)      -> usize    { self.shape.iter().product() }
    pub fn is_empty(&self) -> bool     { self.shape.contains(&0) }
    pub(crate) fn strides(&self)  -> &[usize] { &self.strides }
}

impl PartialEq for RawTensor {
    fn eq(&self, other: &RawTensor) -> bool {
        self.shape == other.shape && 
        self.contiguous_data() == other.contiguous_data()
    }
}
