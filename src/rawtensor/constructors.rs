use super::RawTensor;
use super::structure::strides_contiguous;
use std::rc::Rc;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};

impl RawTensor {
    // Returns a contiguous RawTensor from a shape and a shared reference of the data
    fn new(shape: &[usize], data: Rc<[f64]>) -> RawTensor {
        assert!(shape.iter().all(|&d| d > 0), "dimensions must be non-zero");
        assert_eq!(shape.iter().product::<usize>(), data.len(), "data length doesn't match shape");

        RawTensor {
            shape: Box::from(shape), 
            strides: strides_contiguous(shape),
            data,
        }
    }

    // Returns a contiguous RawTensor from a shape and a reference to an Rc (no copying)
    pub fn from_rc(shape: &[usize], data: &Rc<[f64]>) -> RawTensor {
        RawTensor::new(shape, Rc::clone(data))
    }

    // Returns a contiguous RawTensor from a shape and a Box (no copying)
    pub fn from_box(shape: &[usize], data: Box<[f64]>) -> RawTensor {
        RawTensor::new(shape, Rc::from(data))
    }

    // Returns a contiguous RawTensor from a shape and a vec (no copying)
    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> RawTensor {
        RawTensor::new(shape, Rc::from(data.into_boxed_slice()))
    }

    // Returns a contiguous RawTensor from a shape and a slice (copying data)
    pub fn from_slice(shape: &[usize], data: &[f64]) -> RawTensor {
        RawTensor::new(shape, Rc::from(data))
    }

    // Returns a new RawTensor with data in logical order
    pub fn contiguous(&self) -> RawTensor {
        RawTensor::from_rc(&self.shape, &self.contiguous_data())
    }

    // Creates a RawTensor filled with value
    pub fn full(shape: &[usize], value: f64) -> RawTensor {
        let len: usize = shape.iter().product();
        RawTensor::from_vec(shape, vec![value; len])
    }

    // Creates a RawTensor filled with 0.0.
    pub fn zeros(shape: &[usize]) -> RawTensor {
        RawTensor::full(shape, 0.0)
    }

    // Creates a RawTensor filled with 1.0.
    pub fn ones(shape: &[usize]) -> RawTensor {
        RawTensor::full(shape, 1.0)
    }

    // Creates a 1D RawTensor of n evenly spaced values in [start, end] (inclusive on both ends).
    // If n = 1, returns a shape-[1] tensor containing just start.
    pub fn linspace(start: f64, end: f64, n: usize) -> RawTensor {
        if n == 1 {
            return RawTensor::from_vec(&[1], vec![start]);
        }

        let data: Box<[f64]> = (0..n)
            .map(|i| start + i as f64 * (end - start) / (n - 1) as f64)
            .collect();

        RawTensor::from_box(&[n], data)
    }

    // Creates a RawTensor equal to I_n (the idendity matrix of size [n x n])
    pub fn identity(n: usize) -> RawTensor {
        let data: Box<[f64]> = (0..n * n)
            .map(|i| if i % n == i / n {1.0} else {0.0})
            .collect();
        RawTensor::from_box(&[n, n], data)
    }


    // Creates a RawTensor filled with random samples from U([l, r)).
    pub fn rand_range(shape: &[usize], l: f64, r: f64) -> RawTensor {
        assert!(l < r, "[l, r) should be a non-empty interval");

        let len: usize = shape.iter().product();
        let mut rng = thread_rng();
        let uniform = Uniform::new(l, r);
        let data: Box<[f64]> = (0..len).map(|_| uniform.sample(&mut rng)).collect();
        RawTensor::from_box(shape, data)
    }

    // Creates a RawTensor filled with random samples from U([0, 1)).
    pub fn rand(shape: &[usize]) -> RawTensor {
        RawTensor::rand_range(shape, 0.0, 1.0)
    }

    // Creates a RawTensor filled with random samples from N(mean, std_dev).
    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> RawTensor {
        assert!(std_dev > 0.0, "std_dev should be grater than 0");

        let len: usize = shape.iter().product();
        let mut rng = thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();
        let data: Box<[f64]> = (0..len).map(|_| normal.sample(&mut rng)).collect();
        RawTensor::from_box(shape, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_from_slice() {
        let t = RawTensor::from_slice(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    }


    #[test]
    #[should_panic]
    fn new_from_slice_lens_dont_match() {
        let _t = RawTensor::from_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn new_from_vec() {
        let x: Vec<f64> = vec![1.0; 4];
        let t = RawTensor::from_vec(&[2, 2], x);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn zeros_shape_and_data() {
        let t = RawTensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ones_shape_and_data() {
        let t = RawTensor::ones(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn full_fills_value() {
        let t = RawTensor::full(&[3], 7.0);
        assert_eq!(t.shape(), &[3]);
        assert!(t.data().iter().all(|&x| x == 7.0));
    }

    #[test]
    fn linspace_endpoints() {
        let t = RawTensor::linspace(0.0, 1.0, 6);
        let d = t.data();
        assert!((d[0] - 0.0).abs() < 1e-9);
        assert!((d[1] - 0.2).abs() < 1e-9);
        assert!((d[2] - 0.4).abs() < 1e-9);
        assert!((d[3] - 0.6).abs() < 1e-9);
        assert!((d[4] - 0.8).abs() < 1e-9);
        assert!((d[5] - 1.0).abs() < 1e-9);
    }


    #[test]
    fn rand_in_range() {
        let t = RawTensor::rand(&[100]);
        assert!(t.data().iter().all(|&x| (0.0..1.0).contains(&x))); 
    }

    #[test]
    fn linspace_n1_returns_start() {
        let t = RawTensor::linspace(3.0, 99.0, 1);
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.data(), &[3.0]);
    }

    #[test]
    fn linspace_n2_endpoints_only() {
        let t = RawTensor::linspace(0.0, 1.0, 2);
        assert!((t.data()[0] - 0.0).abs() < 1e-9);
        assert!((t.data()[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic]
    fn zero_dim_panics() {
        let _ = RawTensor::from_slice(&[2, 0, 3], &[]);
    }

    #[test]
    #[should_panic]
    fn rand_range_empty_interval_panics() {
        let _ = RawTensor::rand_range(&[10], 5.0, 5.0);
    }

    #[test]
    fn randn_mean_and_std() {                                    
        let n = 10_000;
        let mean = 2.0;                                          
        let std_dev = 1.5;                                       
        let k = 3.0;
        let t = RawTensor::randn(&[n], mean, std_dev);          
        let data = t.data();                                     
                    
        let sample_mean = data.iter().sum::<f64>() / n as f64;   
        let sample_var = data.iter().map(|x| (x - sample_mean).powi(2)).sum::<f64>() / n as f64;               
        let sample_std = sample_var.sqrt();
                                                                 
        let eps_mean = k * std_dev / (n as f64).sqrt();
        let eps_std = k * std_dev / (2.0 * n as f64).sqrt();
                                                                
        assert!((sample_mean - mean).abs() < eps_mean);          
        assert!((sample_std - std_dev).abs() < eps_std);
    }             
}

