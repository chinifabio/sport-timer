use std::{
    env,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use clap::Parser;
use diesel::{Connection, PgConnection, RunQueryDsl, SelectableHelper};
use opencv::core::{MatTraitConst, MatTraitConstManual};
use pgvector::Vector;
use renoir::prelude::*;

use sport_timer::{models::PersonPosition, python::PythonExt, schema, video::VideoExt};

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

    let ctx = StreamContext::new(config);

    let close = Arc::new(AtomicBool::new(false));
    ctx //.update_layer("cameras")
        .stream_frames(args.camera, Some(close.clone()))
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
        // .update_layer("nodes")
        .python::<(Vec<Vec<f32>>, String)>(include_str!("../main.py"))
        .flat_map(|(embeddings, position)| {
            embeddings
                .into_iter()
                .filter(|e| !e.is_empty())
                .map(move |embeddings| (embeddings, position.clone()))
        })
        // .update_layer("cloud")
        // this must live in the cloud because for security reason
        .map(|(embedding, position)| PersonPosition {
            embeddings: Vector::from(embedding),
            position,
        })
        .for_each({
            let pg_connection = pg_connection.clone();
            move |pp| {
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
