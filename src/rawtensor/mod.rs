mod accumulate;     // in-place gradient accumulation for grad/
mod binary;         // elementwise arithmetic between two RawTensors (+, -, *, /)
mod constructors;   // constructors (from_slice, zeros, randn, linspace, etc)
mod core;           // RawTensor type and basic accessors
mod iter;           // iterators in logical order
mod linalg;         // dot product and matmul
mod fmt;            // Debug and Display
mod normalizations; // normalization along an axis (softmax)
mod reductions;     // reductions (sum, mean, var, std_dev, sum_to_shape)
mod scalar;         // arithmetic between a RawTensor and an f64
mod structure;      // shape operations (contiguous, reshape, transpose, expand, squeeze, unsqueeze)
mod unary;          // elementwise unary operations (exp, ln, relu, etc)

pub use core::RawTensor;
