use ag::ndarray;
use autograd as ag;

pub mod dataset;
pub mod model;
pub mod predictable;
pub mod trainable;
pub mod yolo;

pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;
