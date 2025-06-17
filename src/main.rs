use std::{
    env,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime},
};

use clap::Parser;
use diesel::{Connection, PgConnection, RunQueryDsl, SelectableHelper};
use dlib_face_recognition::*;
use dlib_face_recognition::FaceDetectorCnn;
use image::RgbImage;
use opencv::core::{MatTraitConst, MatTraitConstManual};
use pgvector::Vector;
use renoir::prelude::*;

use sport_timer::{
    models::PersonPosition, schema, tracker::Tracker, video::VideoExt,
};

#[derive(clap::Parser)]
struct Args {
    #[clap(long, short)]
    camera: usize,
    #[clap(long)]
    camera_position: Option<String>,
}

pub fn establish_connection() -> PgConnection {
    dotenv::dotenv().ok();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, args) = RuntimeConfig::from_args();
    config.spawn_remote_workers();

    let mut args = Args::parse_from(args);
    let camera_position = args.camera_position.take();
    let pg_connection = Arc::new(Mutex::new(establish_connection()));

    let cnn_detector = Arc::new(Mutex::new(
        FaceDetectorCnn::default().expect("Error loading Face Detector (CNN).")
    ));

    let landmarks = Arc::new(Mutex::new(
        LandmarkPredictor::default().expect("Error loading Landmark Predictor.")
    ));

    let face_encoder = Arc::new(Mutex::new(
        FaceEncoderNetwork::default().expect("Error loading Face Encoder.")
    ));

    let ctx = StreamContext::new(config);

    let close = Arc::new(AtomicBool::new(false));
    ctx //.update_layer("cameras")
        .stream_frames(args.camera, Some(close.clone()))
        .add_timestamps(
            |_| {
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64
            },
            {
                let mut last_ts = 0;
                move |_, ts| {
                    if ts - last_ts > 5 {
                        last_ts = *ts;
                        Some(last_ts)
                    } else {
                        None
                    }
                }
            },
        )
        .filter_map(move |frame| {
            let bytes = frame.data_bytes().ok()?.to_vec();
            let image = RgbImage::from_vec(frame.cols() as u32, frame.rows() as u32, bytes).unwrap();
            let position = camera_position.as_deref().unwrap_or("Unknown").to_string();
            Some((image, position))
        })
        .flat_map({
            let cnn_detector = cnn_detector.clone();
            let landmarks = landmarks.clone();
            let face_encoder = face_encoder.clone();
            move |(image, pos)| {
                let matrix = ImageMatrix::from_image(&image);

                let locations = cnn_detector.lock().unwrap().face_locations(&matrix);

                let mut landmark_results = Vec::new();
                if !locations.is_empty() {
                    let predictor_locked = landmarks.lock().unwrap();
                    for face_location in locations.iter() {
                        let l = predictor_locked.face_landmarks(&matrix, face_location);
                        landmark_results.push(l);
                    }
                }

                if !landmark_results.is_empty() {
                    let encoder_locked = face_encoder.lock().unwrap();
                    let encodings = encoder_locked.get_face_encodings(&matrix, &landmark_results, 0);
                    
                    encodings.into_iter().map(|x| (x.to_owned(), pos.clone())).collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            }
        })
        .map(move |(embeddings, position)| {
            let now = SystemTime::now();
            (Vec::from(embeddings.as_ref()).into_iter().map(|f| f as f32).collect::<Vec<_>>(), position, now)
        })
        // .update_layer("cloud")
        .map(|(embedding, position, timestamp)| PersonPosition {
            embeddings: Vector::from(embedding),
            position,
            timestamp,
        })
        .rich_map({
            let mut tracker = Tracker::default();
            move |pp| tracker.update(pp)
        })
        .group_by(|(id, _)| *id)
        .batch_mode(BatchMode::timed(1024, Duration::from_millis(100)))
        .window(SessionWindow::new(Duration::from_secs(10)))
        .last()
        .drop_key()
        // this must live in the cloud because for security reason
        .for_each({
            let pg_connection = pg_connection.clone();
            move |(_, pp)| {
                println!("sending item");
                let mut conn = pg_connection.lock().unwrap();
                diesel::insert_into(schema::posper::table)
                    .values(&vec![pp])
                    .returning(PersonPosition::as_returning())
                    .get_result(&mut *conn)
                    .expect("Error saving posper");
            }
        });

    ctx.execute_blocking();

    std::thread::sleep(Duration::from_secs(10));
    close.store(true, Ordering::SeqCst);

    Ok(())
}
