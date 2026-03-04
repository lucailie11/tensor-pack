use crate::Tensor;

impl Tensor {
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        let mut mean: f64 = 0.0;
        for (i, &x) in self.data.iter().enumerate() {
            mean += (x - mean) / (i + 1) as f64;
        }
        mean
    }

    pub fn variance(&self) -> f64 {
        let mut mean: f64 = 0.0;
        let mut variance: f64 = 0.0;
        for (i, &x) in self.data.iter().enumerate() {
            let delta: f64 = x - mean;
            mean += delta / (i + 1) as f64;
            variance += delta * (x - mean);
        }
        variance / self.data.len() as f64
    }

    pub fn mean_and_variance(&self) -> (f64, f64) {
        let mut mean: f64 = 0.0;
        let mut variance: f64 = 0.0;
        for (i, &x) in self.data.iter().enumerate() {
            let delta: f64 = x - mean;
            mean += delta / (i + 1) as f64;
            variance += delta * (x - mean);
        }
        (mean, variance / self.data.len() as f64)
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}
