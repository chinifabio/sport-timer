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
use opencv::core::{MatTraitConst, MatTraitConstManual};
use pgvector::Vector;
use renoir::prelude::*;

use sport_timer::{
    models::PersonPosition, python::PythonExt, schema, tracker::Tracker, video::VideoExt,
};

#[derive(clap::Parser)]
struct Args {
    #[clap(long, short)]
    camera: usize,
    #[clap(long)]
    camera_position: Option<String>,
}

pub fn establish_connection() -> Option<PgConnection> {
    let database_url = env::var("DATABASE_URL").unwrap_or_default();
    PgConnection::establish(&database_url).ok()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, args) = RuntimeConfig::from_args();
    config.spawn_remote_workers();

    let mut args = Args::parse_from(args);
    let camera_position = args.camera_position.take();
    let pg_connection = Arc::new(Mutex::new(establish_connection()));

    let ctx = StreamContext::new(config);

    let close = Arc::new(AtomicBool::new(false));
    ctx.update_layer("cameras")
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
            let shape = vec![
                frame.rows() as usize,
                frame.cols() as usize,
                frame.channels() as usize,
            ];
            let position = camera_position.as_deref().unwrap_or("Unknown").to_string();
            Some((bytes, shape, position))
        })
        .python::<(Vec<Vec<f32>>, String)>(include_str!("../main.py"))
        .flat_map(move |(embeddings, position)| {
            let now = SystemTime::now();
            println!(
                "{}: received frame with {} people",
                now.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                embeddings.len()
            );
            embeddings
                .into_iter()
                .filter(|e| !e.is_empty())
                .map(move |embeddings| (embeddings, position.clone(), now))
        })
        .update_layer("cloud")
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
                let mut lock = pg_connection.lock().unwrap();
                let conn = lock.as_mut().unwrap();
                diesel::insert_into(schema::posper::table)
                    .values(&vec![pp])
                    .returning(PersonPosition::as_returning())
                    .get_result(conn)
                    .expect("Error saving posper");
            }
        });

    ctx.execute_blocking();

    std::thread::sleep(Duration::from_secs(10));
    close.store(true, Ordering::SeqCst);

    Ok(())
}
