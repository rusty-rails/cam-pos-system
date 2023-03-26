use image::DynamicImage;
use imageproc::edges::canny;
use nokhwa::{
    nokhwa_initialize,
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    CallbackCamera,
};
fn main() {
    // only needs to be run on OSX
    nokhwa_initialize(|granted| {
        println!("User said {}", granted);
    });
    let cameras = query(ApiBackend::Auto).unwrap();
    cameras.iter().for_each(|cam| println!("{:?}", cam));

    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let first_camera = cameras.first().unwrap();

    let mut threaded = CallbackCamera::new(first_camera.index().clone(), format, |_| {}).unwrap();
    threaded.open_stream().unwrap();
    threaded.poll_frame().unwrap();
    let frame = threaded.poll_frame().unwrap();
    let image = frame.decode_image::<RgbFormat>().unwrap();
    let gray = DynamicImage::ImageRgb8(image).into_luma8();
    let edges = canny(&gray, 40.0, 120.0);
    edges.save("out/canny.jpg").unwrap();
}
