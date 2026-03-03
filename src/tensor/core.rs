pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: &[usize], data: &[f64]) -> Tensor {
        let new_shape: Vec<usize> = shape.to_vec();
        let new_data: Vec<f64> = data.to_vec();

        Tensor {
            shape: new_shape,
            data: new_data,
        }
    }
}
