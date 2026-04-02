use crate::Tensor;

// Welford's online algorithm: numerically stable single-pass mean and population var.
// More stable than E[X^2] - E[X]^2 for data with large mean.

fn sum(data: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    for &x in data {
        sum += x;
    }
    sum
}

fn mean(data: &[f64]) -> f64 {
    sum(data) / data.len() as f64
}

fn mean_and_var(data: &[f64]) -> (f64, f64) {
    let mut mean: f64 = 0.0;
    let mut var: f64 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let delta: f64 = x - mean;
        mean += delta / (i + 1) as f64;
        var += delta * (x - mean);
    }
    (mean, var / data.len() as f64)
}

fn var(data: &[f64]) -> f64 {
    mean_and_var(data).1
}

fn std_dev(data: &[f64]) -> f64 {
    f64::sqrt(var(data))
}

impl Tensor {
    // Core reduction primitive. Folds along `axis` using f(accumulator, element),
    // initialized to 0.0. The output shape is the input shape with `axis` removed.
    //
    // Row-major layout: element at (outer, a, inner) lives at
    //   outer * axis_size * inner_size + a * inner_size + inner
    //
    // Example: shape [3, 4, 5] reduced on axis 1 → shape [3, 5]

    fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64]) -> f64) -> Tensor {
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
                let vec: Vec<f64> = (0..axis_size)
                    .map(|a| self.data[o * axis_size * inner_size + a * inner_size + i])
                    .collect();
                new_data[o * inner_size + i] = f(&vec)
            }
        }

        Tensor {
            shape: new_shape.into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }

    // Computes the sum of elements along `axis`. Output shape drops that dimension.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, sum)
    }

    // Computes the mean of elements along `axis`. Output shape drops that dimension.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, mean)
    }

    // Computes the var of elements along `axis`. Output shape drops that dimension.
    pub fn var_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, var)
    }

    // Computes the stdandard deviation of elements along `axis`. Output shape drops that dimension.
    pub fn std_dev_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, std_dev)
    }

    
}
