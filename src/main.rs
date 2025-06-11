use std::fs::read_dir;

use image::{GenericImageView, ImageReader};
use renoir::prelude::*;

use sport_timer::python::PythonExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = StreamContext::new_local();

    let images = read_dir("images").expect("Failed to read directory. run `mkdir images`");

    ctx.stream_iter(images)
        .filter_map(|x| x.ok().map(|d| d.path()))
        .shuffle()
        .filter_map(|p| {
            println!("processing {p:?}");
            let image = ImageReader::open(p).ok()?.decode().ok()?;
            let (width, height) = image.dimensions();
            let channels = image.color().channel_count();
            let shape = vec![height as usize, width as usize, channels as usize];
            println!("{shape:?}");
            let bytes = image.into_bytes();
            Some((bytes, shape, "uint8".to_string()))
        })
        .python::<Vec<Vec<f32>>>(include_str!("../main.py"))
        .for_each(|embeddings| println!("there are {} detected people", embeddings.len()));

    ctx.execute_blocking();

    Ok(())
}
