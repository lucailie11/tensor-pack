use super::RawTensor;
use crate::utils::stride::softmax;

// Axis-based operations — functions that are applied independently along one axis
// of the tensor while keeping all other axes fixed.
//
// apply_axis_inplace / apply_axis are the core primitives: they iterate over every
// (outer, inner) pair and call f on the slice of `axis_size` elements that spans
// the target axis at that position.
//
// Defined operations:
//   softmax(axis) / softmax_inplace(axis)

impl RawTensor {
    pub fn apply_axis_inplace(&mut self, axis: usize, f: impl Fn(&mut [f64], usize)) {
        assert!(axis < self.shape.len(), "axis out of bounds");

        let mut outer_size: usize = 1;
        let mut inner_size: usize = 1;
        for i in 0..self.shape.len() {
            if i < axis {
                outer_size *= self.shape[i];
            } else if i > axis {
                inner_size *= self.shape[i];
            }
        }
        let outer_size = outer_size;
        let axis_size = self.shape[axis];
        let inner_size = inner_size;

        for o in 0..outer_size {
            for i in 0..inner_size {
                f(&mut self.data[
                    o * axis_size * inner_size + i..
                    o * axis_size * inner_size + axis_size * inner_size], inner_size);
            }
        }
    }

    pub fn apply_axis(&self, axis: usize, f: impl Fn(&mut [f64], usize)) -> RawTensor {
        assert!(axis < self.shape.len(), "axis out of bounds");
        let mut result = RawTensor::new(&self.shape, &self.data);
        result.apply_axis_inplace(axis, f);
        result
    }

    pub fn softmax_inplace(&mut self, axis: usize) {
        self.apply_axis_inplace(axis, softmax);
    }

    pub fn softmax(&self, axis: usize) -> RawTensor {
        self.apply_axis(axis, softmax)
    }
}
