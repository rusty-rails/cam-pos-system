use std::error::Error;

use image::{ImageBuffer, Rgb};
use serde::{Deserialize, Serialize};

use self::{gif_stream::GifStream, mjpeg_stream::MJpegStream, webcam_stream::WebcamStream};

pub mod gif_stream;
pub mod mjpeg_stream;
pub mod webcam_stream;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoSource {
    Webcam(usize),
    Gif(String),
    MJpeg(String),
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
            VideoSource::Gif(path) => Ok(Box::new(GifStream::new(path)?)),
            VideoSource::MJpeg(path) => Ok(Box::new(MJpegStream::new(path)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_source() {
        //let video_source = VideoSource::Webcam(0);
        //let stream_result = VideoSource::new(video_source).unwrap();
        //assert!(stream_result.is_ok());

        let gif_path = "res/red_train.gif";
        let video_source = VideoSource::Gif(gif_path.to_string());
        let stream_result = VideoSource::new(video_source);
        assert!(stream_result.is_ok());
    }
}
