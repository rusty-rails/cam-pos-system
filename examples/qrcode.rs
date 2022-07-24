fn main() {
    let img = image::open("res/qrcodes.png").unwrap();
    let decoder = bardecoder::default_decoder();

    let results = decoder.decode(&img);
    for result in results {
        match result {
            Ok(data) => println!("{}", data),
            Err(err) => println!("{}", err),
        };
    }
}

#[test]
fn test() {
    let img = image::open("res/qr-marker3.png").unwrap();
    let decoder = bardecoder::default_decoder();
    let results = decoder.decode(&img);
    assert_eq!(results[0].as_ref().unwrap(), "marker3");
}
