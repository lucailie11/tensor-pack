use crate::rawtensor::RawTensor;
use crate::Tensor;

pub struct GradInfo {
    #[allow(dead_code)]
    grad: RawTensor,
    #[allow(dead_code)]
    inputs: Vec<Tensor>,
    #[allow(dead_code)]
    backprop: Option<Box<dyn Fn()>>,
}
