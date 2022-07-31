pub mod bbox;
pub mod coords;
pub mod detection;
pub mod frame;
pub mod object_detection;
pub mod object_finder;
#[cfg(feature = "server")]
pub mod ssd;
pub mod stream;
pub mod tracker;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;
