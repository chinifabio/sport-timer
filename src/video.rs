use std::sync::{Arc, atomic::AtomicBool};

use opencv::videoio::{VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst};
use renoir::{
    block::structure::{BlockStructure, OperatorKind, OperatorStructure}, operator::{Operator, StreamElement}, prelude::Source, Stream, StreamContext
};

#[derive(Debug)]
pub struct VideoSource {
    cam: Option<VideoCapture>,
    close: Arc<AtomicBool>,
}

impl VideoSource {
    fn new(camera_index: usize, close: Arc<AtomicBool>) -> Self {
        let cam = VideoCapture::new(camera_index as i32, opencv::videoio::CAP_ANY).unwrap();
        Self {
            cam: Some(cam),
            close,
        }
    }
}

impl Clone for VideoSource {
    fn clone(&self) -> Self {
        Self {
            cam: None,
            close: self.close.clone(),
        }
    }
}

impl std::fmt::Display for VideoSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VideoSource")
    }
}

impl Operator for VideoSource {
    type Out = opencv::core::Mat;

    fn setup(&mut self, _metadata: &mut renoir::ExecutionMetadata) {
        if self.cam.is_none() {
            panic!("Camera not initialized");
        }

        match VideoCapture::is_opened(self.cam.as_ref().unwrap()) {
            Ok(true) => {}
            Ok(false) => panic!("Camera not opened"),
            Err(e) => panic!("Error opening camera: {}", e),
        }
    }

    fn next(&mut self) -> renoir::operator::StreamElement<Self::Out> {
        if self.close.load(std::sync::atomic::Ordering::SeqCst) {
            return renoir::operator::StreamElement::Terminate;
        }

        let cam = self.cam.as_mut().unwrap();
        let mut frame = opencv::core::Mat::default();
        match cam.read(&mut frame) {
            Ok(true) => StreamElement::Item(frame),
            Ok(false) => {
                eprintln!("No frame read");
                renoir::operator::StreamElement::Terminate
            }
            Err(e) => {
                eprintln!("Error reading frame: {}", e);
                renoir::operator::StreamElement::Terminate
            }
        }
    }

    fn structure(&self) -> BlockStructure {
        let mut operator = OperatorStructure::new::<Self::Out, _>("VideoSource");
        operator.kind = OperatorKind::Source;
        BlockStructure::default().add_operator(operator)
    }
}

impl Source for VideoSource {
    fn replication(&self) -> renoir::Replication {
        renoir::Replication::One
    }
}

pub trait VideoExt {
    fn stream_frames(
        &self,
        cam_index: usize,
        close: Option<Arc<AtomicBool>>,
    ) -> Stream<VideoSource>;
}

impl VideoExt for StreamContext {
    fn stream_frames(
        &self,
        cam_index: usize,
        close: Option<Arc<AtomicBool>>,
    ) -> Stream<VideoSource> {
        let source = VideoSource::new(cam_index, close.unwrap_or_default());
        self.stream(source)
    }
}
