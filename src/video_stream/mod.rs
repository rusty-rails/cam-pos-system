use std::error::Error;

use image::{ImageBuffer, Rgb};
use serde::{Deserialize, Serialize};

use self::webcam_stream::WebcamStream;

pub mod webcam_stream;

#[derive(Debug, Deserialize, Serialize)]
pub enum VideoSource {
    Webcam(usize),
    Gif(String),
}

impl Default for VideoSource {
    fn default() -> Self {
        Self::Webcam(0)
    }
}

pub trait VideoStream: Send {
    fn frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>>;
}

impl VideoSource {
    pub fn new(source: VideoSource) -> Result<Box<dyn VideoStream>, Box<dyn Error>> {
        match source {
            VideoSource::Webcam(index) => Ok(Box::new(WebcamStream::new(index)?)),
            VideoSource::Gif(_) => todo!(),
        }
    }
}
