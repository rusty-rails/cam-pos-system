use gif::Decoder;
use image::{ImageBuffer, Rgb};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;

use super::VideoStream;

pub struct GifStream {
    file: File,
    decoder: Decoder<BufReader<File>>,
}

impl GifStream {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let decoder = Self::create_decoder(&file)?;
        Ok(Self { file, decoder })
    }

    fn create_decoder(file: &File) -> Result<Decoder<BufReader<File>>, Box<dyn Error>> {
        let mut options = gif::DecodeOptions::new();
        options.set_color_output(gif::ColorOutput::RGBA);
        let reader = BufReader::new(file.try_clone()?);
        let decoder = options.read_info(reader)?;
        Ok(decoder)
    }
}

impl VideoStream for GifStream {
    fn frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn Error>> {
        let frame = self.decoder.read_next_frame()?;
        match frame {
            Some(frame) => {
                let (width, height) = (frame.width as u32, frame.height as u32);
                let (left, top) = (frame.left as u32, frame.top as u32);
                let buffer = &frame.buffer;
                let mut image = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(width, height);

                for (x, y, pixel) in image.enumerate_pixels_mut() {
                    let index = ((y * width + x) * 4) as usize;
                    let color = image::Rgb([buffer[index], buffer[index + 1], buffer[index + 2]]);
                    *pixel = color;
                }

                let (canvas_width, canvas_height) =
                    (self.decoder.width() as u32, self.decoder.height() as u32);
                let mut canvas =
                    image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(canvas_width, canvas_height);

                image::imageops::overlay(&mut canvas, &image, left, top);

                Ok(canvas)
            }
            None => {
                self.file.seek(SeekFrom::Start(0))?;
                self.decoder = Self::create_decoder(&self.file)?;
                self.frame()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn gif_stream_test() -> Result<(), Box<dyn Error>> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("res/red_train.gif");

        let mut gif_stream = GifStream::new(path)?;

        for _ in 0..10 {
            let frame = gif_stream.frame()?;
            if frame.width() == 0 && frame.height() == 0 {
                break;
            }
            assert!(frame.width() > 0 && frame.height() > 0);
        }

        Ok(())
    }
}
