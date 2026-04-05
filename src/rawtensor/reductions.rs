use super::RawTensor;
use crate::utils::stride::{sum, mean, var, std_dev};

// Reduction operations along a single axis
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

        RawTensor::from_vec(&new_shape, new_data)
    }

    pub fn sum_axis(&self, axis: usize) -> RawTensor { self.reduce_axis(axis, sum) }
    pub fn mean_axis(&self, axis: usize) -> RawTensor { self.reduce_axis(axis, mean) }
    pub fn var_axis(&self, axis: usize) -> RawTensor { self.reduce_axis(axis, var) }
    pub fn std_dev_axis(&self, axis: usize) -> RawTensor { self.reduce_axis(axis, std_dev) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

    #[test]
    fn sum_axis0() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.sum_axis(0);
        assert_eq!(b.shape(), &[3]);
        assert_eq!(b.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis1() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.sum_axis(1);
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.data(), &[6.0, 15.0]);
    } 

    #[test]
    fn mean_axis0() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.mean_axis(0);
        assert!(b.data().iter().zip(&[2.0, 3.0]).all(|(&x, &y)| approx_eq(x, y)));
    }
}
