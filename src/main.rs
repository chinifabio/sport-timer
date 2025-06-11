use clap::Parser;
use opencv::core::{MatTraitConst, MatTraitConstManual};
use renoir::prelude::*;

use sport_timer::{python::PythonExt, video::VideoExt};

#[derive(clap::Parser)]
struct Args {
    #[clap(long, short)]
    camera: usize
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, args) = RuntimeConfig::from_args();
    config.spawn_remote_workers();

    let args = Args::parse_from(args);

    let ctx = StreamContext::new(config);

    ctx.stream_frames(args.camera, None)
        .filter_map(|frame| {
            let bytes = frame.data_bytes().ok()?.to_vec();
            let shape = vec![frame.rows() as usize, frame.cols() as usize, frame.channels() as usize];
            Some((bytes, shape))
        })
        // .update_requirements(s("python").eq("yes"))
        .python::<Option<Vec<Vec<f32>>>>(include_str!("../main.py"))
        .filter_map(|f| f)
        // .update_requirements(none())
        .for_each(|embeddings| println!("there are {} detected people", embeddings.len()));

    ctx.execute_blocking();

    Ok(())
}
