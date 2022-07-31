use cam_pos_system::coords::Coords;
use cam_pos_system::frame::Frame;
use cam_pos_system::object_detection;
use cam_pos_system::ssd::yolo::Yolo;
use cam_pos_system::stream::MJpeg;
use cam_pos_system::tracker::Tracker;
use image::{DynamicImage, ImageBuffer, Rgb};
use nalgebra::Point2;
use nokhwa::ThreadedCamera;
use rocket::fs::FileServer;
use rocket::http::{ContentType, Status};
use rocket::response::stream::ByteStream;
use rocket::serde::json::Json;
use rocket::State;
use rocket::{get, options, post, routes};
use serde::{Deserialize, Serialize};
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
    (
        Status::Ok,
        (
            ContentType::new("multipart", "x-mixed-replace").with_params([("boundary", "frame")]),
            ByteStream(MJpeg::new(frame_mutex)),
        ),
    )
}

#[get("/yolo")]
fn yolo(
    frame: &'_ State<Arc<Mutex<Frame>>>,
    yolo: &'_ State<Arc<Mutex<Yolo>>>,
) -> (Status, (ContentType, Vec<u8>)) {
    let frame = {
        let base_img: DynamicImage = {
            let img = frame.lock().unwrap();
            DynamicImage::ImageRgb8(img.raw.clone())
        };
        let image = {
            let yolo = yolo.lock().unwrap();
            yolo.run(&base_img)
        };
        let mut buf = vec![];
        image
            .write_to(&mut buf, image::ImageOutputFormat::Jpeg(70))
            .unwrap();
        buf
    };
    (Status::Ok, (ContentType::JPEG, frame))
}

#[derive(Serialize, Deserialize)]
struct ObjectCoords {
    id: u32,
    world_x: f32,
    world_y: f32,
    model_x: f32,
    model_y: f32,
}

#[get("/coords")]
fn get_coords(
    frame: &'_ State<Arc<Mutex<Frame>>>,
    tracker: &'_ State<Arc<Mutex<Tracker>>>,
    coords: &'_ State<Arc<Mutex<Coords<f32>>>>,
) -> Json<Vec<ObjectCoords>> {
    let mut object_coords = Vec::new();
    let predictions = {
        let frame = frame.lock().unwrap();
        let mut tracker = tracker.lock().unwrap();
        tracker.tracker.track(&frame.luma)
    };
    let coords = coords.lock().unwrap();
    for (obj_id, pred) in predictions.iter() {
        let world_x = pred.location.0 as f32;
        let world_y = pred.location.1 as f32;
        let world_point = Point2::new(world_x, world_y);
        let model_point = coords.to_model(&world_point);
        object_coords.push(ObjectCoords {
            id: *obj_id,
            world_x,
            world_y,
            model_x: model_point.x,
            model_y: model_point.y,
        });
    }

    Json(object_coords)
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
    coords: &'_ State<Arc<Mutex<Coords<f32>>>>,
    request: Json<UpdatePositionRequest>,
) {
    let data = request.into_inner();
    let frame = frame.lock().unwrap();
    let mut tracker = tracker.lock().unwrap();
    tracker
        .tracker
        .add_target(data.id, (data.x, data.y), &frame.luma);
    let mut coords = coords.lock().unwrap();
    match data.id {
        1 => coords.set_marker1(Point2::new(data.x as f32, data.y as f32)),
        2 => coords.set_marker2(Point2::new(data.x as f32, data.y as f32)),
        3 => coords.set_marker3(Point2::new(data.x as f32, data.y as f32)),
        _ => (),
    }
}

#[options("/update-position")]
fn options_update_position() {}

async fn fetch_frame(frame: Arc<Mutex<Frame>>, webcam: Arc<Mutex<ThreadedCamera>>) {
    loop {
        {
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
        {
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

    {
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
    let mut tracker = Tracker::new(width, height);
    let loco_detector = object_detection::ObjectDetection::new("res/loco5".to_string());
    {
        let frame = frame.lock().unwrap();
        let prediction = loco_detector.predict(&frame.luma);
        tracker
            .tracker
            .add_target(5, prediction.location, &frame.luma);
    };
    let marker1_detector = object_detection::ObjectDetection::new("res/marker1".to_string());
    {
        let frame = frame.lock().unwrap();
        let prediction = marker1_detector.predict(&frame.luma);
        tracker
            .tracker
            .add_target(1, prediction.location, &frame.luma);
    };
    let marker2_detector = object_detection::ObjectDetection::new("res/marker2".to_string());
    {
        let frame = frame.lock().unwrap();
        let prediction = marker2_detector.predict(&frame.luma);
        tracker
            .tracker
            .add_target(2, prediction.location, &frame.luma);
    };
    let marker3_detector = object_detection::ObjectDetection::new("res/marker3".to_string());
    {
        let frame = frame.lock().unwrap();
        let prediction = marker3_detector.predict(&frame.luma);
        tracker
            .tracker
            .add_target(3, prediction.location, &frame.luma);
    };
    let tracker = Arc::new(Mutex::new(tracker));

    let tracker_thread = tokio::spawn(run_tracker(tracker.clone(), frame.clone()));

    let (w, h) = (width as f32, height as f32);

    let coords = Arc::new(Mutex::new(Coords::new(
        Point2::new(w, 0.0),
        Point2::new(0.0, 0.0),
        Point2::new(0.0, h),
    )));

    let yolo = Arc::new(Mutex::new(Yolo::default()));

    let launcher = rocket::build()
        .mount("/", FileServer::from("static"))
        .mount(
            "/",
            routes![
                frame,
                luma,
                yolo,
                tracked,
                tracked_stream,
                get_coords,
                post_update_position,
                options_update_position
            ],
        )
        .manage(webcam)
        .manage(frame)
        .manage(tracker)
        .manage(coords)
        .manage(yolo);
    let _server = launcher.launch().await.unwrap();
    fetch_frame_thread.abort();
    tracker_thread.abort();
}
