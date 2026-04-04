use super::Tensor;
use super::utils::stride_len;

// Reduction operations along a single tensor axis (sum, mean, variance, std dev).
// Each public method delegates to `reduce_axis`, which handles the strided traversal.

fn sum(data: &[f64], step: usize) -> f64 {
    data.iter().step_by(step).sum()
}

fn mean(data: &[f64], step: usize) -> f64 {
    sum(data, step) / stride_len(data, step) as f64
}

// Computes mean and population variance in a single pass using Welford's online algorithm.
fn mean_and_var(data: &[f64], step: usize) -> (f64, f64) {
    let mut mean: f64 = 0.0;
    let mut var: f64 = 0.0;
    for (i, x) in data.iter().step_by(step).enumerate() {
        let delta: f64 = x - mean;
        mean += delta / (i + 1) as f64;
        var += delta * (x - mean);
    }
    (mean, var / stride_len(data, step) as f64)
}

fn var(data: &[f64], step: usize) -> f64 {
    mean_and_var(data, step).1
}

fn std_dev(data: &[f64], step: usize) -> f64 {
    f64::sqrt(var(data, step))
}

impl Tensor {
    // Core reduction primitive. For each (outer, inner) lane, passes a strided slice
    // starting at the first axis element and a step size (inner_size) to f, which reads
    // every step-th value to cover the axis. The output shape is the input shape with `axis` removed.
    //
    // Example: shape [3, 4, 5] reduced on axis 1 → shape [3, 5]

    fn reduce_axis(&self, axis: usize, f: impl Fn(&[f64], usize) -> f64) -> Tensor {
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

    // Computes the standard deviation of elements along `axis`. Output shape drops that dimension.
    pub fn std_dev_axis(&self, axis: usize) -> Tensor {
        self.reduce_axis(axis, std_dev)
    }
}
