// Returns the number of elements reachable from `data` when stepping by `step`.
// Equivalent to ceil(data.len() / step).
fn stride_len(data: &[f64], step: usize) -> usize {
    (data.len() - 1) / step + 1
}

pub(crate) fn sum(data: &[f64], step: usize) -> f64 {
    data.iter().step_by(step).sum()
}

pub(crate) fn mean(data: &[f64], step: usize) -> f64 {
    sum(data, step) / stride_len(data, step) as f64
}

pub(crate) fn mean_and_var(data: &[f64], step: usize) -> (f64, f64) {
    let mut mean: f64 = 0.0;
    let mut var: f64 = 0.0;
    for (i, x) in data.iter().step_by(step).enumerate() {
        let delta: f64 = x - mean;
        mean += delta / (i + 1) as f64;
        var += delta * (x - mean);
    }
    (mean, var / stride_len(data, step) as f64)
}

pub(crate) fn var(data: &[f64], step: usize) -> f64 {
    mean_and_var(data, step).1
}

pub(crate) fn std_dev(data: &[f64], step: usize) -> f64 {
    f64::sqrt(var(data, step))
}

pub(crate) fn softmax(data: &mut [f64], step: usize) {
    let max: f64 = data
        .iter()
        .step_by(step)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    for x in data.iter_mut().step_by(step) {
        *x -= max;
    }

    let sum: f64 = data
        .iter()
        .step_by(step)
        .map(|x| f64::exp(*x))
        .sum();

    for x in data.iter_mut().step_by(step) {
        *x = f64::exp(*x) / sum;
    }
}


