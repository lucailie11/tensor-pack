use super::RawTensor;
use crate::utils::stride::{sum, mean, var, std_dev};

use std::rc::Rc;

// Reduction operations along a single axis
// Core primitive is reduce_axis
//
// Defined operations
// - sum
// - mean
// - var
// - std_dev

impl RawTensor {
    pub fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64], usize, usize) -> f64) -> RawTensor {
        assert!(axis < self.shape.len(), "axis out of bounds");

        let n = self.shape[axis];
        let new_shape: Box<[usize]> = self.shape.iter().enumerate().
            filter(|(i, _)| *i != axis).map(|(_, x)| *x).collect();

        if self.strides[axis] == 0 {
            let view_shape: Box<[usize]> = new_shape.clone();
            let view_strides: Box<[usize]> = self.strides.iter().enumerate().
                filter(|(i, _)| *i != axis).map(|(_, x)| *x).collect();

            let view = RawTensor {
                shape: view_shape,
                strides: view_strides,
                data: Rc::clone(&self.data),
            };

            let new_data: Box<[f64]> = view.iter()
                .map(|x| {
                    let repeated: Box<[f64]> = vec![x; n].into_boxed_slice();
                    f(&repeated, 1, n)
                })
                .collect();

            return RawTensor::from_box(&new_shape, new_data);
        }

        let new_data: Box<[f64]> = self.iter().enumerate()
            .filter(|(i, _)| (i / self.strides[axis]).is_multiple_of(self.shape[axis]))
            .map(|(i, _)| f(&self.data[i..i + (n - 1) * self.strides[axis] + 1], self.strides[axis], n))
            .collect();

        RawTensor::from_box(&new_shape, new_data)
    }

    //TODO
    pub fn reduce(&self, _f: impl Fn(&[f64], usize, usize) -> f64) -> f64 {
        0.0
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

    #[test]
    fn var_axis0() {
        // var([1,3]) = 1.0, var([2,4]) = 1.0
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.var_axis(0);
        assert_eq!(b.shape(), &[2]);
        assert!(b.data().iter().zip(&[1.0, 1.0]).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    fn std_dev_axis0() {
        let a = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = a.std_dev_axis(0);
        assert_eq!(b.shape(), &[2]);
        assert!(b.data().iter().zip(&[1.0, 1.0]).all(|(&x, &y)| approx_eq(x, y)));
    }

    #[test]
    fn var_axis1_known_values() {
        // row [2, 4, 6]: mean=4, var = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = 8/3
        let a = RawTensor::from_slice(&[1, 3], &[2.0, 4.0, 6.0]);
        let b = a.var_axis(1);
        assert_eq!(b.shape(), &[1]);
        assert!(approx_eq(b.data()[0], 8.0 / 3.0));
    }

    #[test]
    fn sum_axis_3d() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let a = RawTensor::from_vec(&[2, 3, 4], data);
        let b = a.sum_axis(1);
        assert_eq!(b.shape(), &[2, 4]);
        assert_eq!(b.data()[0], 0.0 + 4.0 + 8.0);
        assert_eq!(b.data()[1], 1.0 + 5.0 + 9.0);
    }

    #[test]
    #[should_panic]
    fn reduce_axis_out_of_bounds() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0; 6]);
        let _ = a.sum_axis(2);
    }

    #[test]
    fn reduce_transposed() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.transpose(&[1, 0]);
        let c = b.sum_axis(0);
        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.data(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_expanded() {
        let a = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.expand(&[5, 2, 3]);
        let c = b.sum_axis(0);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), &[5.0, 10.0, 15.0, 20.0, 25.0, 30.0]);
    }
}
