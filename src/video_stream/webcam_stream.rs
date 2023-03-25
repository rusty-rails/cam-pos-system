use image::ImageBuffer;
use image::Rgb;
use nokhwa::ThreadedCamera;
use std::error::Error;

use super::VideoStream;

pub struct WebcamStream {
    cam: ThreadedCamera,
}

impl WebcamStream {
    pub fn new(index: usize) -> Result<WebcamStream, Box<dyn Error>> {
        let cam = ThreadedCamera::new(index, None).unwrap();

        let mut stream = WebcamStream { cam: cam };
        stream.cam.open_stream(Self::callback)?;
        Ok(stream)
    }

    fn callback(_image: ImageBuffer<Rgb<u8>, Vec<u8>>) {}
}

impl VideoStream for WebcamStream {
    fn frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        let frame = self.cam.poll_frame()?;
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "needs webcam"]
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
