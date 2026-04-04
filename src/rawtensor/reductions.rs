use super::RawTensor;
use crate::utils::stride::{sum, mean, var, std_dev};

// Reduction operations along a single tensor axis (sum, mean, variance, std dev).
// Each public method delegates to `reduce_axis`, which handles the strided traversal.

impl RawTensor {
    // Core reduction primitive. For each (outer, inner) lane, passes a strided slice
    // starting at the first axis element and a step size (inner_size) to f, which reads
    // every step-th value to cover the axis. The output shape is the input shape with `axis` removed.
    //
    // Example: shape [3, 4, 5] reduced on axis 1 → shape [3, 5]

    pub fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64], usize) -> f64) -> RawTensor {
        assert!(axis < self.shape.len(), "axis out of bounds");

        let mut outer_size: usize = 1;
        let mut inner_size: usize = 1;
        let mut new_shape: Vec<usize> = Vec::with_capacity(self.shape.len() - 1);
        for i in 0..self.shape.len() {
            if i != axis {
                new_shape.push(self.shape[i]);
            }

            if i < axis {
                outer_size *= self.shape[i];
            } else if i > axis {
                inner_size *= self.shape[i];
            }
        }
        let new_shape = new_shape;
        let outer_size = outer_size;
        let axis_size = self.shape[axis];
        let inner_size = inner_size;

        let mut new_data: Vec<f64> = vec![0.0; outer_size * inner_size];
        for o in 0..outer_size {
            for i in 0..inner_size {
                new_data[o * inner_size + i] = f(&self.data[
                    o * axis_size * inner_size + i..
                    o * axis_size * inner_size + axis_size * inner_size], inner_size);
            }
        }

        RawTensor {
            shape: new_shape.into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }

    // Computes the sum of elements along `axis`. Output shape drops that dimension.
    pub fn sum_axis(&self, axis: usize) -> RawTensor {
        self.reduce_axis(axis, sum)
    }

    // Computes the mean of elements along `axis`. Output shape drops that dimension.
    pub fn mean_axis(&self, axis: usize) -> RawTensor {
        self.reduce_axis(axis, mean)
    }

    // Computes the var of elements along `axis`. Output shape drops that dimension.
    pub fn var_axis(&self, axis: usize) -> RawTensor {
        self.reduce_axis(axis, var)
    }

    // Computes the standard deviation of elements along `axis`. Output shape drops that dimension.
    pub fn std_dev_axis(&self, axis: usize) -> RawTensor {
        self.reduce_axis(axis, std_dev)
    }
}
