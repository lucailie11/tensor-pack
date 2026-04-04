use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use super::RawTensor;

impl RawTensor {
    // All constructors take the desired shape as a slice of dimension sizes.
    // Data is stored in row-major order: the last axis varies fastest.

    // Creates a RawTensor from existing shape and data slices. Panics if their sizes are inconsistent.
    pub fn from_slice(shape: &[usize], data: &[f64]) -> RawTensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Data length doesn't match shape"
        );

        RawTensor {
            shape: Box::from(shape),
            data: Box::from(data),
        }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> RawTensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Data length doesn't match shape"
        );

        RawTensor {
            shape: Box::from(shape),
            data: data.into_boxed_slice(),
        }
    }

    pub fn from_box(shape: &[usize], data: Box<[f64]>) -> RawTensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Data length doesn't match shape"
        );

        RawTensor {
            shape: Box::from(shape),
            data,
        }
    }

    // Changes the shape without moving data. The total number of elements must stay the same.
    pub fn reshape(&mut self, new_shape: &[usize]) {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.data.len(),
            "New shape doesn't match old data length"
        );
        self.shape = Box::from(new_shape);
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
            return RawTensor {
                shape: vec![1].into_boxed_slice(),
                data: vec![start].into_boxed_slice(),
            };
        }

        let data: Vec<f64> = (0..n)
            .map(|i| start + i as f64 * (end - start) / (n - 1) as f64)
            .collect();

        RawTensor::from_vec(&[n], data)
    }

    // Creates a RawTensor filled with samples from U([l, r)).
    pub fn rand_range(shape: &[usize], l: f64, r: f64) -> RawTensor {
        assert!(l < r, "[l, r) should be a non-empty interval");
        let len: usize = shape.iter().product();
        let mut rng = thread_rng();
        let uniform = Uniform::new(l, r);
        let data: Vec<f64> = (0..len).map(|_| uniform.sample(&mut rng)).collect();

        RawTensor::from_vec(shape, data)
    }

    // Creates a RawTensor filled with samples from U([0, 1)).
    pub fn rand(shape: &[usize]) -> RawTensor {
        RawTensor::rand_range(shape, 0.0, 1.0)
    }

    // Creates a RawTensor filled with samples from N(mean, std_dev).
    pub fn randn(shape: &[usize], mean: f64, std_dev: f64) -> RawTensor {
        let len: usize = shape.iter().product();
        let mut rng = thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();
        let data: Vec<f64> = (0..len).map(|_| normal.sample(&mut rng)).collect();

        RawTensor::from_vec(shape, data)
    }
}
