use image::ImageBuffer;
use image::Rgb;
use nokhwa::utils::CameraIndex;
use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    CallbackCamera,
};
use std::error::Error;

use super::VideoStream;

pub struct WebcamStream {
    cam: CallbackCamera,
}

impl WebcamStream {
    pub fn new(index: usize) -> Result<WebcamStream, Box<dyn Error>> {
        nokhwa_initialize(|granted| {
            println!("Camera access granted {}", granted);
        });
        let cameras = query(ApiBackend::Auto).unwrap();
        cameras.iter().for_each(|cam| println!("{:?}", cam));
        let index = CameraIndex::Index(index as u32);
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
        let cam = CallbackCamera::new(index, format, |_| {})?;

        let mut stream = WebcamStream { cam: cam };
        stream.cam.open_stream()?;
        Ok(stream)
    }
}

impl VideoStream for WebcamStream {
    fn frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        let frame = self.cam.poll_frame()?;
        Ok(frame.decode_image::<RgbFormat>().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //#[ignore = "needs webcam"]
    #[test]
    fn test_webcam_stream() -> Result<(), Box<dyn Error>> {
        let index = 0;
        let mut webcam_stream = WebcamStream::new(index)?;

        let frame = webcam_stream.frame()?;

        assert!(frame.width() > 0);
        assert!(frame.height() > 0);

        Ok(())
    }
}
