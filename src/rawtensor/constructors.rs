use super::RawTensor;
use super::structure::strides_contiguous;

use std::rc::Rc;
use std::cell::RefCell;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

// A set of basic tensor constructors
// Constructors take desired shape as a parameter unless specified otherwise
// and return contiguous RawTensors

impl RawTensor {
    // Returns a contiguous RawTensor from a reference to an Rc (no copying)
    pub fn from_rc(shape: &[usize], data: Rc<[f64]>) -> RawTensor {
        assert!(shape.iter().all(|&d| d > 0), "dimensions must be non-zero");
        assert_eq!(shape.iter().product::<usize>(), data.len(), "data length doesn't match shape");

        RawTensor {
            shape: Box::from(shape), 
            strides: strides_contiguous(shape),
            data,
        }
    }

    // Returns a contiguous RawTensor from a Box (copies data)
    pub fn from_box(shape: &[usize], data: Box<[f64]>) -> RawTensor {
        RawTensor::from_rc(shape, Rc::from(data))
    }

    // Returns a contiguous RawTensor from a Vec (copies data)
    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> RawTensor {
        RawTensor::from_rc(shape, Rc::from(data))
    }

    // Returns a contiguous RawTensor from a slice (copies data)
    pub fn from_slice(shape: &[usize], data: &[f64]) -> RawTensor {
        RawTensor::from_rc(shape, Rc::from(data))
    }

    // Returns a 0D RawTensor from a scalar (f64)
    pub fn from_scalar(scalar: f64) -> RawTensor {
        RawTensor::from_slice(&[], &[scalar])
    }

    // Returns a new RawTensor with data in logical order
    pub fn contiguous(&self) -> RawTensor {
        RawTensor::from_rc(&self.shape, self.contiguous_data())
    }

    // Creates a RawTensor filled with value
    pub fn full(shape: &[usize], value: f64) -> RawTensor {
        let len: usize = shape.iter().product();
        RawTensor::from_vec(shape, vec![value; len])
    }

    // Creates a RawTensor filled with 0.0
    pub fn zeros(shape: &[usize]) -> RawTensor {
        RawTensor::full(shape, 0.0)
    }

    // Creates a RawTensor filled with 1.0
    pub fn ones(shape: &[usize]) -> RawTensor {
        RawTensor::full(shape, 1.0)
    }

    // Creates a 1D RawTensor of n evenly spaced values in [start, end] (inclusive on both ends).
    // If n = 1, returns a shape-[1] tensor containing just start
    pub fn linspace(start: f64, end: f64, n: usize) -> RawTensor {
        assert_ne!(n, 0, "can't have a 0 dimenstion");

        if n == 1 {
            return RawTensor::from_vec(&[1], vec![start]);
        }

        let data: Rc<[f64]> = (0..n)
            .map(|i| start + i as f64 * (end - start) / (n - 1) as f64)
            .collect();

        RawTensor::from_rc(&[n], data)
    }

    // Creates a RawTensor equal to I_n (the identity matrix of size [n x n])
    pub fn identity(n: usize) -> RawTensor {
        let data: Rc<[f64]> = (0..n * n)
            .map(|i| if i % n == i / n {1.0} else {0.0})
            .collect();
        RawTensor::from_rc(&[n, n], data)
    }


    pub fn set_seed(seed: u64) {
        RNG.with(|rng| *rng.borrow_mut() = StdRng::seed_from_u64(seed));
    }

    // Creates a RawTensor filled with random samples from U([l, r))
    pub fn rand_range(shape: &[usize], l: f64, r: f64) -> RawTensor {
        assert!(l.is_finite() && r.is_finite(), "l and r must be finite");
        assert!(l < r, "[l, r) should be a non-empty interval");

        let len: usize = shape.iter().product();
        let uniform = Uniform::new(l, r);
        let data: Rc<[f64]> = RNG.with(|rng| {
            (0..len).map(|_| uniform.sample(&mut *rng.borrow_mut())).collect()
        });
        RawTensor::from_rc(shape, data)
    }

    // Creates a RawTensor filled with random samples from U([0, 1))
    pub fn rand(shape: &[usize]) -> RawTensor {
        RawTensor::rand_range(shape, 0.0, 1.0)
    }

    // Creates a RawTensor filled with random samples from N(mean, std_dev)
    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> RawTensor {
        assert!(mean.is_finite(), "mean must be finite");
        assert!(std_dev > 0.0 && std_dev.is_finite(), "std_dev must be finite and greater than 0");

        let len: usize = shape.iter().product();
        let normal = Normal::new(mean, std_dev).unwrap();
        let data: Rc<[f64]> = RNG.with(|rng| {
            (0..len).map(|_| normal.sample(&mut *rng.borrow_mut())).collect()
        });
        RawTensor::from_rc(shape, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_shape_and_data() {
        let t = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(*t.shape, [2, 3]);
        assert_eq!(*t.contiguous_data(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn from_slice_shape_mismatch_panics() {
        let _ = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic]
    fn zero_dim_panics() {
        let _ = RawTensor::from_slice(&[2, 0, 3], &[]);
    }

    #[test]
    fn from_scalar_is_0d() {
        let t = RawTensor::from_scalar(7.0);
        assert_eq!(*t.shape, []);
        assert_eq!(t.ndim(), 0);
        assert_eq!(*t.contiguous_data(), [7.0]);
    }

    #[test]
    fn zeros_shape_and_data() {
        let t = RawTensor::zeros(&[2, 3]);
        assert_eq!(*t.shape, [2, 3]);
        assert!(t.iter().all(|x| x == 0.0));
    }

    #[test]
    fn ones_shape_and_data() {
        let t = RawTensor::ones(&[2, 3]);
        assert_eq!(*t.shape, [2, 3]);
        assert!(t.iter().all(|x| x == 1.0));
    }

    #[test]
    fn full_fills_value() {
        let t = RawTensor::full(&[2, 3], 7.0);
        assert_eq!(*t.shape, [2, 3]);
        assert!(t.iter().all(|x| x == 7.0));
    }

    #[test]
    fn linspace_values_and_length() {
        let t = RawTensor::linspace(0.0, 5.0, 6);
        assert_eq!(*t.shape, [6]);
        assert_eq!(&*t.contiguous_data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = RawTensor::linspace(3.0, 99.0, 1);
        assert_eq!(*t.shape, [1]);
        assert_eq!(&*t.contiguous_data(), &[3.0]);
    }

    #[test]
    fn identity_diagonal_and_off_diagonal() {
        let t = RawTensor::identity(3);
        assert_eq!(t.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t.get(&[i, j]), Some(expected));
            }
        }
    }

    #[test]
    fn rand_in_unit_range() {
        let t = RawTensor::rand(&[1000]);
        assert!(t.iter().all(|x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn rand_range_in_bounds() {
        let t = RawTensor::rand_range(&[1000], -2.0, 5.0);
        assert!(t.iter().all(|x| (-2.0..5.0).contains(&x)));
    }

    #[test]
    fn seed_reproducibility() {
        RawTensor::set_seed(21);
        let a = RawTensor::rand_range(&[6], 0.0, 5.0);

        RawTensor::set_seed(52);
        let b = RawTensor::rand_range(&[6], 0.0, 5.0);

        RawTensor::set_seed(21);
        let c = RawTensor::rand_range(&[6], 0.0, 5.0);

        assert_eq!(a.contiguous_data(), c.contiguous_data());
        assert_ne!(a.contiguous_data(), b.contiguous_data());
    }


    #[test]
    #[should_panic]
    fn rand_range_empty_interval_panics() {
        let _ = RawTensor::rand_range(&[10], 5.0, 5.0);
    }

    #[test]
    #[should_panic]
    fn rand_range_inf_l_panics() {
        let _ = RawTensor::rand_range(&[10], f64::NEG_INFINITY, 5.0);
    }

    #[test]
    #[should_panic]
    fn rand_range_inf_r_panics() {
        let _ = RawTensor::rand_range(&[10], 0.0, f64::INFINITY);
    }

    #[test]
    #[should_panic]
    fn randn_zero_std_dev_panics() {
        let _ = RawTensor::randn(&[10], 5.0, 0.0);
    }

    #[test]
    #[should_panic]
    fn randn_inf_std_dev_panics() {
        let _ = RawTensor::randn(&[10], 0.0, f64::INFINITY);
    }

    #[test]
    #[should_panic]
    fn randn_nan_mean_panics() {
        let _ = RawTensor::randn(&[10], f64::NAN, 1.0);
    }

    #[test]
    #[should_panic]
    fn randn_inf_mean_panics() {
        let _ = RawTensor::randn(&[10], f64::INFINITY, 1.0);
    }

    #[test]
    fn randn_mean_and_std() {
        let n = 10_000;
        let mean = 2.0;
        let std_dev = 1.5;
        let k = 3.0;
        let data: Vec<f64> = RawTensor::randn(&[n], mean, std_dev).iter().collect();

        let sample_mean = data.iter().sum::<f64>() / n as f64;
        let sample_var = data.iter().map(|x| (x - sample_mean).powi(2)).sum::<f64>() / n as f64;
        let sample_std = sample_var.sqrt();

        assert!((sample_mean - mean).abs() < k * std_dev / (n as f64).sqrt());
        assert!((sample_std - std_dev).abs() < k * std_dev / (2.0 * n as f64).sqrt());
    }

    #[test]
    fn contiguous_on_transposed() {
        let t = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).transpose(&[1, 0]);
        assert!(!t.is_contiguous());
        let c = t.contiguous();
        assert!(c.is_contiguous());
        assert_eq!(*c.shape, [3, 2]);
        assert_eq!(*c.strides, [2, 1]);
        assert_eq!(*c.contiguous_data(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn contiguous_on_expanded() {
        let t = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]).expand(&[4, 3]);
        assert!(!t.is_contiguous());
        let c = t.contiguous();
        assert!(c.is_contiguous());
        assert_eq!(*c.shape, [4, 3]);
        assert_eq!(*c.strides, [3, 1]);
        assert_eq!(*c.contiguous_data(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
}

