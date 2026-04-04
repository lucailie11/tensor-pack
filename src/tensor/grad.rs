use crate::rawtensor::RawTensor;
use super::Tensor;

pub struct GradInfo {
    grad: RawTensor,
    inputs: Vec<Tensor>,
    backprop: Option<Box<dyn Fn()>>,
}
