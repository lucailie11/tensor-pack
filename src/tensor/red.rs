use crate::Tensor;

// Welford's online algorithm: numerically stable single-pass mean and population variance.
// More stable than E[X^2] - E[X]^2 for data with large mean.
fn mean_and_variance(data: &[f64]) -> (f64, f64) {
    let mut mean: f64 = 0.0;
    let mut variance: f64 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let delta: f64 = x - mean;
        mean += delta / (i + 1) as f64;
        variance += delta * (x - mean);
    }
    (mean, variance / data.len() as f64)
}

impl Tensor {
    // Core reduction primitive. Folds along `axis` using f(accumulator, element),
    // initialized to 0.0. The output shape is the input shape with `axis` removed.
    //
    // Row-major layout: element at (outer, a, inner) lives at
    //   outer * axis_size * inner_size + a * inner_size + inner
    //
    // Example: shape [3, 4, 5] reduced on axis 1 → shape [3, 5]
    fn reduce_axis(&self, axis: usize, f: impl Fn(f64, f64) -> f64) -> Tensor {
        assert!(!self.shape.is_empty(), "reduce_axis requires at least a 1D tensor");
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
            for a in 0..axis_size {
                for i in 0..inner_size {
                    new_data[o * inner_size + i] = f(
                        new_data[o * inner_size + i],
                        self.data[o * axis_size * inner_size + a * inner_size + i],
                    );
                }
            }
        }

        Tensor {
            shape: new_shape.into_boxed_slice(),
            data: new_data.into_boxed_slice(),
        }
    }

    // Sums elements along `axis`. Output shape drops that dimension.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, |acc, x| acc + x)
    }

    // TODO: implement full reductions (sum, mean, variance, std_dev)
}
