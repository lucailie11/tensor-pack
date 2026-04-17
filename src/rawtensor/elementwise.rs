use super::RawTensor;
use super::broadcasting::{get_broadcast_index, get_broadcast_shape};

impl RawTensor {
    // Panics if the shapes are incompatible for broadcasting
    pub fn elementwise_op(&self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) -> RawTensor {
        let out_shape = get_broadcast_shape(&self.shape, &other.shape);
        let out_len: usize = out_shape.iter().product();
        let new_data: Vec<f64> = (0..out_len)
            .map(|i| f(
                self.data[get_broadcast_index(i, &self.shape, &out_shape)],
                other.data[get_broadcast_index(i, &other.shape, &out_shape)],
            ))
            .collect();

        RawTensor {
            shape: out_shape.into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }

    // Panics if self doesn't already have the broadcast output shape (other is broadcast into self)
    pub fn elementwise_op_inplace(&mut self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) {
        assert_eq!(
            get_broadcast_shape(&self.shape, &other.shape).as_slice(),
            &*self.shape,
            "in-place broadcasting requires self to already be the output shape"
        );
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i], other.data[get_broadcast_index(i, &other.shape, &self.shape)]);
        }
    }

    // does not broadcast
    pub fn elementwise_op_inplace_3(&mut self, first: &RawTensor, second: &RawTensor, f: impl Fn(f64, f64, f64) -> f64) {
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i], 
                first.data[i],
                second.data[i],
            );
        }
    }

    // does not broadcast
    pub fn elementwise_op_inplace_4(&mut self, first: &RawTensor, second: &RawTensor, third: &RawTensor, f: impl Fn(f64, f64, f64, f64) -> f64) {
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i], 
                first.data[i],
                second.data[i],
                third.data[i],
            );
        }
    }
}


