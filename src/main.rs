use cam_pos_system::frame::Frame;
use image::{DynamicImage, ImageBuffer, Rgb};
use nokhwa::ThreadedCamera;
use rocket::http::{ContentType, Status};
use rocket::State;
use rocket::{get, routes};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn callback(_image: ImageBuffer<Rgb<u8>, Vec<u8>>) {}

#[get("/frame")]
fn frame(frame: &'_ State<Arc<Mutex<Frame>>>) -> (Status, (ContentType, Vec<u8>)) {
    let frame = {
        let img = frame.lock().unwrap();
        let base_img: DynamicImage = DynamicImage::ImageRgb8(img.raw.clone());
        let mut buf = vec![];
        base_img
            .write_to(&mut buf, image::ImageOutputFormat::Jpeg(70))
            .unwrap();
        buf
    };
    (Status::Ok, (ContentType::JPEG, frame))
}

#[get("/luma")]
fn luma(frame: &'_ State<Arc<Mutex<Frame>>>) -> (Status, (ContentType, Vec<u8>)) {
    let frame = {
        let img = frame.lock().unwrap();
        let base_img: DynamicImage = DynamicImage::ImageLuma8(img.luma.clone());
        let mut buf = vec![];
        base_img
            .write_to(&mut buf, image::ImageOutputFormat::Jpeg(70))
            .unwrap();
        buf
    };
    (Status::Ok, (ContentType::JPEG, frame))
}

async fn fetch_frame(frame: Arc<Mutex<Frame>>, webcam: Arc<Mutex<ThreadedCamera>>) {
    loop {
        let image = webcam.lock().unwrap().poll_frame().unwrap();
        let mut frame = frame.lock().unwrap();
        *frame = Frame::new(image);
        thread::sleep(Duration::from_millis(30));
    }
}

#[tokio::main]
async fn main() {
    let frame = Arc::new(Mutex::new(Frame::default()));

    let mut webcam = ThreadedCamera::new(0, None).unwrap();
    webcam.open_stream(callback).unwrap();

    let webcam = Arc::new(Mutex::new(webcam));

    let fetch_frame_thread = tokio::spawn(fetch_frame(frame.clone(), webcam.clone()));

    let launcher = rocket::build()
        .mount("/", routes![frame, luma])
        .manage(webcam)
        .manage(frame);
    let _server = launcher.launch().await.unwrap();
    fetch_frame_thread.abort();
}
