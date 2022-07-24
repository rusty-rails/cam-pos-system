use image::{DynamicImage, ImageBuffer, Luma, Rgb};

#[derive(Default)]
pub struct Frame {
    pub raw: ImageBuffer<Rgb<u8>, Vec<u8>>,
    pub luma: ImageBuffer<Luma<u8>, Vec<u8>>,
}

impl Frame {
    pub fn new(raw: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Frame {
        let luma = DynamicImage::ImageRgb8(raw.clone()).to_luma8();
        Frame { raw, luma }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default() {
        let img = image::open("res/qr-marker3.png").unwrap();
        let frame = Frame::new(img.to_rgb8());
        assert_eq!(frame.luma, img.to_luma8());
    }
}
