pub fn sum(data: &[f64], step: usize, n: usize) -> f64 {
    if n == 0 { return 0.0; }
    if step == 0 { return data[0] * n as f64; }
    data.iter().step_by(step).take(n).sum()
}

pub fn mean(data: &[f64], step: usize, n: usize) -> f64 {
    if n == 0 { return 0.0; }
    sum(data, step, n) / n as f64
}

pub fn mean_and_var(data: &[f64], step: usize, n: usize) -> (f64, f64) {
    if n == 0 { return (0.0, 0.0); }
    if step == 0 { return (data[0], 0.0); }
    let mut mean: f64 = 0.0;
    let mut var: f64 = 0.0;
    for (i, x) in data.iter().step_by(step).take(n).enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        var += delta * (x - mean);
    }
    (mean, var / n as f64)
}

pub fn var(data: &[f64], step: usize, n: usize) -> f64 {
    mean_and_var(data, step, n).1
}

pub fn std_dev(data: &[f64], step: usize, n: usize) -> f64 {
    f64::sqrt(var(data, step, n))
}

pub fn softmax(data: &mut [f64], step: usize, n: usize) {
    if n == 0 { return; }
    let max: f64 = data.iter().step_by(step).take(n).copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = data.iter().step_by(step).take(n).map(|&x| f64::exp(x - max)).collect();
    let sum: f64 = exps.iter().sum();
    for (x, e) in data.iter_mut().step_by(step).take(n).zip(exps) {
        *x = e / sum;
    }
}
