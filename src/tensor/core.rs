pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn full(shape: &[usize], value: f64) -> Tensor {
        let len: usize = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: vec![value; len],
        }
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn linspace(start: f64, end: f64, n: usize) -> Tensor {
        assert!(n > 1, "n must be at least 2");

        let lin_data: Vec<f64> = (0..n)
            .map(|i| start + i as f64 * (end - start) / (n - 1) as f64)
            .collect();

        Tensor {
            shape: vec![n],
            data: lin_data,
        }
    }

    pub fn new(shape: &[usize], data: &[f64]) -> Tensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Data length doesn't match shape"
        );

        Tensor {
            shape: shape.to_vec(),
            data: data.to_vec(),
        }
    }
}
