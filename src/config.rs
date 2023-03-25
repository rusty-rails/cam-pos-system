use crate::video_stream::VideoSource;
use figment::{
    providers::{Env, Format, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct Config {
    pub video_source: VideoSource,
}

pub fn load_config() -> Config {
    Figment::new()
        .merge(Toml::file(Env::var_or("ROCKET_CONFIG", "Rocket.toml")).nested())
        .merge(Env::prefixed("ROCKET_").ignore(&["PROFILE"]).global())
        .extract()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::env;

    fn setup_rocket_toml() -> String {
        let content = r#"
            [default]
            video_source = { webcam = 0 }
        "#;

        let rocket_toml = std::env::temp_dir().join("Rocket.toml");
        std::fs::write(&rocket_toml, content).expect("Unable to create Rocket.toml file");
        rocket_toml
            .into_os_string()
            .into_string()
            .expect("Failed to get Rocket.toml path")
    }

    #[test]
    fn test_load_config() {
        let rocket_toml_path = setup_rocket_toml();
        env::set_var("ROCKET_CONFIG", &rocket_toml_path);

        let config = load_config();

        match config.video_source {
            VideoSource::Webcam(index) => {
                assert_eq!(index, 0);
            }
            _ => panic!("Expected video_source to be a Webcam variant"),
        }

        std::fs::remove_file(rocket_toml_path).expect("Unable to delete Rocket.toml file");
    }
}
