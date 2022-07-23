use image::{ImageBuffer, Rgb};
use nokhwa::{query_devices, CaptureAPIBackend, ThreadedCamera};

fn main() {
    let cameras = query_devices(CaptureAPIBackend::Auto).unwrap();
    cameras.iter().for_each(|cam| println!("{:?}", cam));

    let mut threaded = ThreadedCamera::new(0, None).unwrap();
    threaded.open_stream(callback).unwrap();

    let frame = threaded.poll_frame().unwrap();
    frame.save("out/webcam.jpg").unwrap();
}

fn callback(image: ImageBuffer<Rgb<u8>, Vec<u8>>) {
    println!("{}x{} {}", image.width(), image.height(), image.len());
}
