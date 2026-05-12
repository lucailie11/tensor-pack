mod accumulate;     // in-place gradient accumulation for grad/
mod core;           // RawTensor type and basic accessors
mod iter;           // iterators in logical order
mod binary;         // elementwise arithmetic between two RawTensors (+, -, *, /)
mod constructors;   // constructors (from_slice, zeros, randn, linspace, ...)
mod linalg;         // dot product and matmul
mod fmt;            // Debug and Display
mod normalizations; // normalization along an axis (softmax, ...)
mod reductions;     // reductions along an axis (sum, mean, var, std_dev, ...)
mod scalar;         // arithmetic between a RawTensor and an f64
mod structure;      // shape operations (reshape, transpose, expand, squeeze, unsqueeze, ...)
mod unary;          // elementwise unary operations (exp, ln, relu, ...)

pub(crate) use core::RawTensor;
