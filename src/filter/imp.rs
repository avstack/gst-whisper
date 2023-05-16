use std::{env, iter, mem::MaybeUninit, sync::{Arc, atomic::{AtomicBool, Ordering}, mpsc, Mutex}, thread, time::Instant};

use byte_slice_cast::AsSliceOf;
use gstreamer::glib::{self, ParamSpec, ParamSpecBoolean, SignalHandlerId, Value};
use gstreamer::{
  element_imp_error, loggable_error,
  param_spec::GstParamSpecBuilderExt,
  prelude::{Cast, ElementExt, GstBinExt, GstObjectExt, ObjectExt, PadExt, ParamSpecBuilderExt, ToValue},
  subclass::{
    prelude::{ElementImpl, ElementImplExt, GstObjectImpl, ObjectImpl, ObjectSubclass, ObjectSubclassExt},
    ElementMetadata,
  },
  Buffer, BufferFlags, Caps, CapsIntersectMode, ClockTime, CoreError, DebugCategory, ErrorMessage,
  Event, EventView, FlowError, FlowSuccess, LoggableError, Message, PadDirection, PadPresence, PadTemplate, Pipeline,
};
use gstreamer_audio::{AudioCapsBuilder, AudioInfo, AudioLayout, AUDIO_FORMAT_S16, prelude::BaseTransformExtManual};
use gstreamer_base::{
  subclass::{
    base_transform::{BaseTransformImpl, BaseTransformImplExt, GenerateOutputSuccess},
    BaseTransformMode,
  },
  BaseTransform,
};
use once_cell::sync::Lazy;
use webrtc_vad::{Vad, VadMode};
use whisper_rs::{convert_integer_to_float_audio, FullParams, SamplingStrategy, WhisperContext, WhisperState};

const SAMPLE_RATE: usize = 16_000;

static WHISPER_CONTEXT: Lazy<WhisperContext> = Lazy::new(|| {
  let path = env::var("WHISPER_MODEL_PATH").unwrap();
  WhisperContext::new(&path).unwrap()
});

static CAT: Lazy<DebugCategory> = Lazy::new(|| {
  DebugCategory::new(
    "whisper",
    gstreamer::DebugColorFlags::empty(),
    Some("Speech to text filter using Whisper"),
  )
});

static SINK_CAPS: Lazy<Caps> = Lazy::new(|| {
  AudioCapsBuilder::new()
    .format(AUDIO_FORMAT_S16)
    .layout(AudioLayout::NonInterleaved)
    .rate(SAMPLE_RATE as i32)
    .channels(1)
    .build()
});

static SRC_CAPS: Lazy<Caps> =
  Lazy::new(|| Caps::builder("text/x-raw").field("format", "utf8").build());

#[derive(Debug, Clone, Default)]
struct Settings {}

struct State {
  whisper_state: WhisperState<'static>,
  voice_activity_detected: Arc<AtomicBool>,
  vad_sender: mpsc::Sender<Vec<i16>>,
  chunk: Option<Chunk>,
  prev_buffer: Vec<i16>,
}

struct Chunk {
  start_pts: ClockTime,
  buffer: Vec<i16>,
}

pub struct WhisperFilter {
  settings: Mutex<Settings>,
  state: Mutex<Option<State>>,
}

impl WhisperFilter {
  fn whisper_params(&self) -> FullParams {
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_single_segment(true);
    params.set_language(Some("en"));
    params.set_suppress_blank(true);
    params.set_no_speech_thold(0.3);
    // params.set_no_context(true);
    // params.set_tokens(&state.tokens);
    params
  }

  fn run_model(&self, state: &mut State, chunk: Chunk) -> Result<Option<Buffer>, FlowError> {
    gstreamer::debug!(CAT, "run_model()");

    let start = Instant::now();
    let samples = convert_integer_to_float_audio(&chunk.buffer);
    gstreamer::debug!(CAT, "run_model(): sample conversion took {:?}", start.elapsed());

    let start = Instant::now();
    state.whisper_state.full(self.whisper_params(), &samples).unwrap();
    gstreamer::debug!(CAT, "run_model(): model took {:?}", start.elapsed());

    let n_segments = state.whisper_state.full_n_segments().unwrap();
    if n_segments > 0 {
      assert!(n_segments == 1);

      let segment = state
        .whisper_state
        .full_get_segment_text(0)
        .unwrap()
        .replace("[BLANK_AUDIO]", "")
        .replace("[ Silence ]", "")
        .replace("[silence]", "")
        .replace("(silence)", "")
        .replace("[ Pause ]", "")
        .trim()
        .to_owned();
      
      if !segment.is_empty() {
        let start_ts = state.whisper_state.full_get_segment_t0(0).unwrap();
        let end_ts = state.whisper_state.full_get_segment_t1(0).unwrap();

        gstreamer::debug!(
          CAT,
          "run_model(): {} - {}: {}",
          start_ts,
          end_ts,
          segment
        );

        let segment = format!("{}\n", segment);
        let mut buffer = Buffer::with_size(segment.len()).map_err(|_| FlowError::Error)?;
        let buffer_mut = buffer.get_mut().ok_or(FlowError::Error)?;
        buffer_mut.set_pts(chunk.start_pts.checked_add(ClockTime::from_mseconds(start_ts as u64 * 10)).unwrap());
        buffer_mut.set_duration(ClockTime::from_mseconds((end_ts as u64 - start_ts as u64) * 10));
        buffer_mut
          .copy_from_slice(0, segment.as_bytes())
          .map_err(|_| FlowError::Error)?;

        // state.tokens = WHISPER_CONTEXT.tokenize(&segment, state.whisper_state.full_n_tokens(0).map_err(|_| FlowError::Error)? as usize).map_err(|_| FlowError::Error)?;

        Ok(Some(buffer))
      }
      else {
        gstreamer::debug!(CAT, "run_model(): empty segment");
        Ok(None)
      }
    }
    else {
      gstreamer::debug!(CAT, "run_model(): no segment");
      Ok(None)
    }
  }
}

#[glib::object_subclass]
impl ObjectSubclass for WhisperFilter {
  const NAME: &'static str = "GstWhisperFilter";
  type Type = super::WhisperFilter;
  type ParentType = BaseTransform;

  fn new() -> Self {
    Self {
      settings: Mutex::new(Default::default()),
      state: Mutex::new(None),
    }
  }
}

impl ObjectImpl for WhisperFilter {
  fn properties() -> &'static [ParamSpec] {
    static PROPERTIES: Lazy<Vec<ParamSpec>> = Lazy::new(|| vec![]);
    PROPERTIES.as_ref()
  }

  fn set_property(&self, _id: usize, value: &Value, pspec: &ParamSpec) {
    match pspec.name() {
      _ => unimplemented!(),
    }
  }

  fn property(&self, _id: usize, pspec: &ParamSpec) -> Value {
    match pspec.name() {
      name => panic!("No getter for {name}"),
    }
  }
}

impl GstObjectImpl for WhisperFilter {}

impl ElementImpl for WhisperFilter {
  fn metadata() -> Option<&'static ElementMetadata> {
    static ELEMENT_METADATA: Lazy<ElementMetadata> = Lazy::new(|| {
      ElementMetadata::new(
        "Transcriber",
        "Audio/Text/Filter",
        "Speech to text filter using Whisper",
        "Jasper Hugo <jasper@avstack.io>",
      )
    });

    Some(&*ELEMENT_METADATA)
  }

  fn pad_templates() -> &'static [PadTemplate] {
    static PAD_TEMPLATES: Lazy<Vec<PadTemplate>> = Lazy::new(|| {
      let src_pad_template =
        PadTemplate::new("src", PadDirection::Src, PadPresence::Always, &SRC_CAPS).unwrap();

      let sink_pad_template = gstreamer::PadTemplate::new(
        "sink",
        gstreamer::PadDirection::Sink,
        gstreamer::PadPresence::Always,
        &SINK_CAPS,
      )
      .unwrap();

      vec![src_pad_template, sink_pad_template]
    });

    PAD_TEMPLATES.as_ref()
  }
}

impl BaseTransformImpl for WhisperFilter {
  const MODE: BaseTransformMode = BaseTransformMode::NeverInPlace;
  const PASSTHROUGH_ON_SAME_CAPS: bool = false;
  const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

  fn start(&self) -> Result<(), ErrorMessage> {
    gstreamer::debug!(CAT, "start()");

    let voice_activity_detected = Arc::new(AtomicBool::new(false));
    let (vad_sender, vad_receiver) = mpsc::channel::<Vec<i16>>();
    {
      let voice_activity_detected = voice_activity_detected.clone();
      thread::spawn(move || {
        gstreamer::debug!(CAT, "vad starting");
        let mut vad = Vad::new_with_rate_and_mode((SAMPLE_RATE as i32).try_into().unwrap(), VadMode::Aggressive);
        while let Ok(next) = vad_receiver.recv() {
          let result = vad.is_voice_segment(&next).unwrap();
          gstreamer::debug!(CAT, "vad result: {}", result);
          voice_activity_detected.store(result, Ordering::Relaxed);
        }
        gstreamer::debug!(CAT, "vad stopped");
      });
    }

    *self.state.lock().unwrap() = Some(State {
      whisper_state: WHISPER_CONTEXT.create_state().unwrap(),
      voice_activity_detected,
      vad_sender,
      chunk: None,
      prev_buffer: Vec::new(),
    });

    gstreamer::debug!(CAT, "start(): started");
    Ok(())
  }

  fn stop(&self) -> Result<(), ErrorMessage> {
    gstreamer::debug!(CAT, "stop()");
    if let Some(state) = self.state.lock().unwrap().take() {
      // if let Some(bus) = self.obj().bus() {
      //   bus.disconnect(state.bus_message_handler_id);
      // }
      // else {
      //   gstreamer::debug!(CAT, "stop(): bus went away; message handler may be dangling");
      // }
    }
    gstreamer::debug!(CAT, "stop(): stopped");
    Ok(())
  }

  fn transform_caps(
    &self,
    direction: PadDirection,
    _caps: &Caps,
    maybe_filter: Option<&Caps>,
  ) -> Option<Caps> {
    gstreamer::debug!(CAT, "transform_caps({:?})", direction);
    let mut caps = if direction == PadDirection::Src {
      SINK_CAPS.clone()
    } else {
      SRC_CAPS.clone()
    };
    if let Some(filter) = maybe_filter {
      caps = filter.intersect_with_mode(&caps, CapsIntersectMode::First);
    }
    Some(caps)
  }

  fn generate_output(&self) -> Result<GenerateOutputSuccess, FlowError> {
    gstreamer::debug!(CAT, "generate_output()");
    if let Some(buffer) = self.take_queued_buffer() {
      gstreamer::debug!(CAT, "generate_output(): got queued buffer");

      let mut locked_state = self.state.lock().unwrap();
      let state = locked_state.as_mut().ok_or_else(|| {
        element_imp_error!(
          self,
          CoreError::Negotiation,
          ["Can not generate an output without state"]
        );
        FlowError::NotNegotiated
      })?;
      gstreamer::debug!(CAT, "generate_output(): locked state");
  
      let buffer_reader = buffer
        .as_ref()
        .map_readable()
        .map_err(|_| FlowError::Error)?;
      let samples = buffer_reader
        .as_slice_of()
        .map_err(|_| FlowError::Error)?;
      gstreamer::debug!(CAT, "generate_output(): reading {} samples", samples.len());

      let buffer_len = samples.len();
      if buffer_len >= 160 {
        let vad_buffer = if buffer_len >= 480 {
          &samples[buffer_len-480..buffer_len]
        }
        else if buffer_len >= 320 {
          &samples[buffer_len-320..buffer_len]
        }
        else {
          &samples[buffer_len-160..buffer_len]
        };
        state.vad_sender.send(vad_buffer.to_vec()).unwrap();
      }

      if state.voice_activity_detected.load(Ordering::Relaxed) {
        if state.chunk.is_none() {
          gstreamer::debug!(CAT, "generate_output(): voice activity started");
          state.chunk = Some(Chunk {
            start_pts: buffer.pts().unwrap(),
            buffer: state.prev_buffer.drain(..).collect(),
          });
        }
        state.chunk.as_mut().unwrap().buffer.extend_from_slice(samples);
  
        gstreamer::debug!(CAT, "generate_output(): unlocked state");
        Ok(GenerateOutputSuccess::NoOutput)
      }
      else {
        state.prev_buffer = samples.to_vec();

        if let Some(chunk) = state.chunk.take() {
          gstreamer::debug!(CAT, "generate_output(): voice activity ended");
          let maybe_buffer = self.run_model(state, chunk)?;
          gstreamer::debug!(CAT, "generate_output(): unlocked state");
          Ok(maybe_buffer.map(GenerateOutputSuccess::Buffer).unwrap_or(GenerateOutputSuccess::NoOutput))
        }
        else {
          gstreamer::debug!(CAT, "generate_output(): unlocked state");
          Ok(GenerateOutputSuccess::NoOutput)
        }
      }
    }
    else {
      gstreamer::debug!(CAT, "generate_output(): no queued buffers to take");
      Ok(GenerateOutputSuccess::NoOutput)
    }
  }
}
