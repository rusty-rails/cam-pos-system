use cam_pos_system::frame::Frame;
use cam_pos_system::stream::MJpeg;
use cam_pos_system::tracker::Tracker;
use image::{DynamicImage, ImageBuffer, Rgb};
use nokhwa::ThreadedCamera;
use rocket::fs::FileServer;
use rocket::http::{ContentType, Status};
use rocket::response::stream::ByteStream;
use rocket::serde::json::Json;
use rocket::State;
use rocket::{get, options, post, routes};
use serde::Deserialize;
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

#[get("/tracked")]
fn tracked(frame: &'_ State<Arc<Mutex<Frame>>>) -> (Status, (ContentType, Vec<u8>)) {
    let frame = {
        let img = frame.lock().unwrap();
        let base_img: DynamicImage = DynamicImage::ImageRgb8(img.tracked.clone());
        let mut buf = vec![];
        base_img
            .write_to(&mut buf, image::ImageOutputFormat::Jpeg(70))
            .unwrap();
        buf
    };
    (Status::Ok, (ContentType::JPEG, frame))
}

#[get("/tracked-stream")]
fn tracked_stream(
    frame: &'_ State<Arc<Mutex<Frame>>>,
) -> (Status, (ContentType, ByteStream<MJpeg>)) {
    let frame_mutex: Arc<Mutex<Frame>> = { frame.inner().clone() };
    drop(frame);
    (
        Status::Ok,
        (
            ContentType::new("multipart", "x-mixed-replace").with_params([("boundary", "frame")]),
            ByteStream(MJpeg::new(frame_mutex.clone())),
        ),
    )
}

#[derive(Deserialize)]
struct UpdatePositionRequest {
    id: u32,
    x: u32,
    y: u32,
}

#[post("/update-position", format = "application/json", data = "<request>")]
fn post_update_position(
    frame: &'_ State<Arc<Mutex<Frame>>>,
    tracker: &'_ State<Arc<Mutex<Tracker>>>,
    request: Json<UpdatePositionRequest>,
) {
    let data = request.into_inner();
    println!("{}, {}, {}", data.id, data.x, data.y);
    let frame = frame.lock().unwrap();
    let mut tracker = tracker.lock().unwrap();
    tracker
        .tracker
        .add_target(data.id, (data.x, data.y), &frame.luma);
}

#[options("/update-position")]
fn options_update_position() {}

async fn fetch_frame(frame: Arc<Mutex<Frame>>, webcam: Arc<Mutex<ThreadedCamera>>) {
    loop {
        let _ = {
            let image = webcam.lock().unwrap().last_frame();
            let mut frame = frame.lock().unwrap();
            let mut f = Frame::new(image);
            f.tracked = frame.tracked.clone();
            *frame = f;
        };
        thread::sleep(Duration::from_millis(40));
    }
}

async fn run_tracker(tracker: Arc<Mutex<Tracker>>, frame: Arc<Mutex<Frame>>) {
    loop {
        let _ = {
            let mut frame = frame.lock().unwrap();
            let mut tracker = tracker.lock().unwrap();
            let tracked = tracker.next(&frame.luma);
            frame.tracked = tracked.to_rgb8();
        };
        thread::sleep(Duration::from_millis(25));
    }
}

#[tokio::main]
async fn main() {
    let frame = Arc::new(Mutex::new(Frame::default()));

    let mut webcam = ThreadedCamera::new(0, None).unwrap();
    webcam.open_stream(callback).unwrap();

    let _ = {
        let image = webcam.poll_frame().unwrap();
        let mut frame = frame.lock().unwrap();
        *frame = Frame::new(image);
    };

    let webcam = Arc::new(Mutex::new(webcam));

    let fetch_frame_thread = tokio::spawn(fetch_frame(frame.clone(), webcam.clone()));

    let (width, height) = {
        let frame = frame.lock().unwrap();
        (frame.raw.width(), frame.raw.height())
    };
    let tracker = Arc::new(Mutex::new(Tracker::new(width, height)));

    let tracker_thread = tokio::spawn(run_tracker(tracker.clone(), frame.clone()));

    let launcher = rocket::build()
        .mount("/", FileServer::from("static"))
        .mount(
            "/",
            routes![
                frame,
                luma,
                tracked,
                tracked_stream,
                post_update_position,
                options_update_position
            ],
        )
        .manage(webcam)
        .manage(frame)
        .manage(tracker);
    let _server = launcher.launch().await.unwrap();
    fetch_frame_thread.abort();
    tracker_thread.abort();
}
