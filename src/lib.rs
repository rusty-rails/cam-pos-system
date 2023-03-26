pub mod bbox;
#[cfg(not(target_arch = "wasm32"))]
pub mod config;
pub mod coords;
pub mod dataset;
pub mod detection;
pub mod detector;
pub mod frame;
pub mod object_detection;
pub mod object_finder;
#[cfg(feature = "server")]
pub mod ssd;
pub mod stream;
pub mod tracker;
#[cfg(not(target_arch = "wasm32"))]
pub mod video_stream;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

pub use detector::Detector;
