use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use super::VideoStream;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

pub struct MJpegStream {
    reader: BufReader<File>,
    current_frame: Option<DynamicImage>,
}

impl MJpegStream {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut soi_marker = [0u8; 2];
        reader.read_exact(&mut soi_marker)?;
        if soi_marker != [0xFF, 0xD8] {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "The file does not appear to be an MJPEG file",
            )));
        }
        reader.seek(SeekFrom::Start(0))?;

        let mut stream = Self {
            reader,
            current_frame: None,
        };
        stream.advance_frame()?;
        Ok(stream)
    }

    fn advance_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let mut buf = Vec::new();
        let mut inside_frame = false;
        loop {
            let mut byte = [0u8; 1];
            if self.reader.read_exact(&mut byte).is_err() {
                self.reader.seek(SeekFrom::Start(0))?;
                break;
            }

            if byte == [0xFF] {
                let mut next_byte = [0u8; 1];
                if self.reader.read_exact(&mut next_byte).is_err() {
                    self.reader.seek(SeekFrom::Start(0))?;
                    break;
                }

                if !inside_frame && next_byte == [0xD8] {
                    inside_frame = true;
                    buf.extend_from_slice(&byte);
                    buf.extend_from_slice(&next_byte);
                } else if inside_frame && next_byte == [0xD9] {
                    buf.extend_from_slice(&byte);
                    buf.extend_from_slice(&next_byte);
                    break;
                } else {
                    buf.extend_from_slice(&byte);
                    buf.extend_from_slice(&next_byte);
                }
            } else {
                buf.extend_from_slice(&byte);
            }
        }
        self.current_frame = Some(image::load_from_memory_with_format(
            &buf,
            ImageFormat::Jpeg,
        )?);
        Ok(())
    }
}

impl VideoStream for MJpegStream {
    fn frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        let rgb_image = match self.current_frame.as_ref() {
            Some(frame) => Ok(frame.to_rgb8()),
            None => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No current frame",
            ))),
        }?;
        if let Err(_) = self.advance_frame() {}
        Ok(rgb_image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn mp4_stream_test() -> Result<(), Box<dyn Error>> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("res/red_train.mjpeg");

        let mut videors_stream = MJpegStream::new(path)?;

        for _ in 0..10 {
            let frame = videors_stream.frame()?;
            if frame.width() == 0 && frame.height() == 0 {
                break;
            }
            assert!(frame.width() > 0 && frame.height() > 0);
        }

        Ok(())
    }
}
