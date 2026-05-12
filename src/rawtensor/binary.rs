use super::RawTensor;
use super::structure::broadcast_shape;

use std::ops::{Add, Sub, Mul, Div};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign};

use std::rc::Rc;

// Binary elementwise operations on two RawTensors with broadcasting support
// Core primitive is elementwise_op
// Assign operations replace the lhs with the result (not in-place)
//
// Defined operations:
//   &RawTensor + &RawTensor  -> RawTensor       RawTensor += &RawTensor
//   &RawTensor - &RawTensor  -> RawTensor       RawTensor -= &RawTensor
//   &RawTensor * &RawTensor  -> RawTensor       RawTensor *= &RawTensor
//   &RawTensor / &RawTensor  -> RawTensor       RawTensor /= &RawTensor

impl RawTensor {
    // Panics if the shapes are incompatible for broadcasting
    // Returns a contiguous RawTensor with a fresh data allocation
    pub fn elementwise_op(&self, other: &RawTensor, f: impl Fn(f64, f64) -> f64) -> RawTensor {
        let out_shape: Box<[usize]> = broadcast_shape(&self.shape, &other.shape).expect("shapes not broadcastable");
        let a = self.expand(&out_shape);
        let b = other.expand(&out_shape);
        let new_data: Rc<[f64]> = a.iter().zip(b.iter())
            .map(|(x, y)| f(x, y))
            .collect(); 
        RawTensor::from_rc(&out_shape, new_data)
    }
}


impl Add for &RawTensor {
    type Output = RawTensor;

    fn add(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a + b)
    }
}

impl Sub for &RawTensor {
    type Output = RawTensor;

    fn sub(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a - b)
    }
}

impl Mul for &RawTensor {
    type Output = RawTensor;

    fn mul(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a * b)
    }
}
impl Div for &RawTensor {
    type Output = RawTensor;

    fn div(self, other: &RawTensor) -> RawTensor {
        self.elementwise_op(other, |a, b| a / b)
    }
}

impl AddAssign<&RawTensor> for RawTensor {
    fn add_assign(&mut self, other: &RawTensor) {
        *self = &*self + other;
    }
}

impl SubAssign<&RawTensor> for RawTensor {
    fn sub_assign(&mut self, other: &RawTensor) {
        *self = &*self - other;
    }
}

impl MulAssign<&RawTensor> for RawTensor {
    fn mul_assign(&mut self, other: &RawTensor) {
        *self = &*self * other;
    }
}

impl DivAssign<&RawTensor> for RawTensor {
    fn div_assign(&mut self, other: &RawTensor) {
        *self = &*self / other;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let a = RawTensor::linspace(1.0, 4.0, 4);
        let b = RawTensor::from_slice(&[4], &[4.0, 3.0, 2.0, 1.0]);
        assert_eq!(*(&a + &b).contiguous_data(), [5.0, 5.0, 5.0, 5.0]);
        assert_eq!(*(&b - &a).contiguous_data(), [3.0, 1.0, -1.0, -3.0]);
        assert_eq!(*(&a * &b).contiguous_data(), [4.0, 6.0, 6.0, 4.0]);
        let c = RawTensor::from_slice(&[4], &[8.0, 6.0, 4.0, 2.0]);
        let d = RawTensor::from_slice(&[4], &[2.0, 3.0, 2.0, 1.0]);
        assert_eq!(*(&c / &d).contiguous_data(), [4.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn assign_ops() {
        let a = RawTensor::from_slice(&[3], &[6.0, 4.0, 2.0]);
        let b = RawTensor::full(&[3], 2.0);
        let mut t = RawTensor::linspace(1.0, 3.0, 3);
        t += &a;
        assert_eq!(*t.contiguous_data(), [7.0, 6.0, 5.0]);
        t -= &b;
        assert_eq!(*t.contiguous_data(), [5.0, 4.0, 3.0]);
        t *= &b;
        assert_eq!(*t.contiguous_data(), [10.0, 8.0, 6.0]);
        t /= &b;
        assert_eq!(*t.contiguous_data(), [5.0, 4.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn incompatible_first_dims_panics() {
        let a = RawTensor::zeros(&[3, 4]);
        let b = RawTensor::zeros(&[4, 4]);
        a.elementwise_op(&b, |x, y| x + y);
    }

    #[test]
    #[should_panic]
    fn incompatible_last_dim_panics() {
        let a = RawTensor::zeros(&[2, 3]);
        let b = RawTensor::zeros(&[2, 4]);
        a.elementwise_op(&b, |x, y| x + y);
    }

    #[test]
    #[should_panic]
    fn incompatible_non_broadcastable_batch_panics() {
        let a = RawTensor::zeros(&[3, 2]);
        let b = RawTensor::zeros(&[1, 4]);
        a.elementwise_op(&b, |x, y| x + y);
    }

    #[test]
    fn non_contiguous_add() {
        let raw1 = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let a = raw1.transpose(&[1, 0]);
        let a = (&a * 2.0).transpose(&[1, 0]);
        let a = -&a;

        let raw2 = RawTensor::linspace(2.0, 12.0, 6).reshape(&[2, 3]);
        let raw2_t = raw2.transpose(&[1, 0]);
        let b = raw2_t.transpose(&[1, 0]);

        assert_eq!(*(&a + &b).shape, [2, 3]);
        assert_eq!(*(&a + &b).strides, [3, 1]);
        assert_eq!(*(&a + &b).contiguous_data(), [0.0; 6]);
    }

    #[test]
    fn non_contiguous_mul_sub() {
        let raw = RawTensor::linspace(1.0, 9.0, 9).reshape(&[3, 3]);
        let raw_t = raw.transpose(&[1, 0]);
        let a_scaled = &raw_t * 3.0;
        let a = &a_scaled + 1.0;
        let raw_scaled = &raw * 2.0;
        let b = &raw_scaled - 1.0;

        assert_eq!(*(&a * &b).contiguous_data(), [4.0, 39.0, 110.0, 49.0, 144.0, 275.0, 130.0, 285.0, 476.0]);
    }

    #[test]
    fn broadcast_row_to_matrix() {
        let a = RawTensor::linspace(1.0, 3.0, 3).reshape(&[1, 3]);
        let b = RawTensor::linspace(10.0, 60.0, 6).reshape(&[2, 3]);
        assert_eq!(*(&a + &b).contiguous_data(), [11.0, 22.0, 33.0, 41.0, 52.0, 63.0]);
    }

    #[test]
    fn broadcast_col_times_row() {
        let a = RawTensor::linspace(1.0, 3.0, 3).reshape(&[3, 1]);
        let b = RawTensor::linspace(4.0, 6.0, 3).reshape(&[1, 3]);
        assert_eq!(*(&a * &b).contiguous_data(), [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]);
    }

    #[test]
    fn non_contiguous_broadcast_add() {
        let raw = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let a = raw.transpose(&[1, 0]);
        let b = RawTensor::from_slice(&[1, 2], &[10.0, 20.0]);
        assert_eq!(*(&a + &b).contiguous_data(), [11.0, 24.0, 12.0, 25.0, 13.0, 26.0]);
    }

    #[test]
    fn expand_add() {
        let a = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]).expand(&[4, 3]);
        let b = RawTensor::linspace(0.0, 11.0, 12).reshape(&[4, 3]);
        assert_eq!(*(&a + &b).contiguous_data(), [1.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 11.0, 10.0, 12.0, 14.0]);
    }

    #[test]
    fn transpose_expand_broadcast_add() {
        let raw = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let a = raw.transpose(&[1, 0]).expand(&[4, 3, 2]);
        let b = RawTensor::linspace(10.0, 40.0, 4).reshape(&[4, 1, 1]);
        let result = &a + &b;
        assert_eq!(*result.shape, [4, 3, 2]);
        assert_eq!(
            *result.contiguous_data(),
            [
                11.0, 14.0, 12.0, 15.0, 13.0, 16.0,
                21.0, 24.0, 22.0, 25.0, 23.0, 26.0,
                31.0, 34.0, 32.0, 35.0, 33.0, 36.0,
                41.0, 44.0, 42.0, 45.0, 43.0, 46.0,
            ]
        );
    }

    #[test]
    fn transpose_broadcast_mul_add_chain() {
        let raw = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]);
        let a = raw.transpose(&[1, 0]);
        let col = RawTensor::from_slice(&[3, 1], &[1.0, 2.0, 3.0]);
        let row = RawTensor::from_slice(&[1, 2], &[10.0, 100.0]);
        let result = &(&a * &col) + &row;
        assert_eq!(*result.shape, [3, 2]);
        assert_eq!(*result.contiguous_data(), [11.0, 104.0, 14.0, 110.0, 19.0, 118.0]);
    }
}
