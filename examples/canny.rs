use image::{ImageBuffer,DynamicImage, Rgb};
use nokhwa::{query_devices, CaptureAPIBackend, ThreadedCamera};
use imageproc::edges::canny;

fn main() {
    let cameras = query_devices(CaptureAPIBackend::Auto).unwrap();
    cameras.iter().for_each(|cam| println!("{:?}", cam));

    let mut threaded = ThreadedCamera::new(0, None).unwrap();
    threaded.open_stream(callback).unwrap();
    threaded.poll_frame().unwrap();
    let frame = threaded.poll_frame().unwrap();
    let gray = DynamicImage::ImageRgb8(frame.try_into().unwrap()).into_luma8();
    let edges = canny(&gray, 40.0, 120.0);
    edges.save("out/canny.jpg").unwrap();
}

fn callback(_image: ImageBuffer<Rgb<u8>, Vec<u8>>) {
}
