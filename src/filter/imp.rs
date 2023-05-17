use std::{
  env,
  sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
  },
  thread,
  time::Instant,
};

use byte_slice_cast::AsSliceOf;
use gstreamer::{
  element_imp_error,
  glib::{self, ParamSpec, Value},
  param_spec::GstParamSpecBuilderExt,
  prelude::{ParamSpecBuilderExt, ToValue},
  subclass::{
    prelude::{ElementImpl, GstObjectImpl, ObjectImpl, ObjectSubclass, ObjectSubclassExt},
    ElementMetadata,
  },
  Buffer, Caps, CapsIntersectMode, ClockTime, CoreError, DebugCategory, ErrorMessage, FlowError,
  PadDirection, PadPresence, PadTemplate,
};
use gstreamer_audio::{AudioCapsBuilder, AudioLayout, AUDIO_FORMAT_S16};
use gstreamer_base::{
  subclass::{
    base_transform::{BaseTransformImpl, BaseTransformImplExt, GenerateOutputSuccess},
    BaseTransformMode,
  },
  BaseTransform,
};
use once_cell::sync::Lazy;
use webrtc_vad::{Vad, VadMode};
use whisper_rs::{
  convert_integer_to_float_audio, FullParams, SamplingStrategy, WhisperContext, WhisperState,
};

const SAMPLE_RATE: usize = 16_000;

const DEFAULT_VAD_MODE: &str = "quality";
const DEFAULT_MIN_VOICE_ACTIVITY_MS: u64 = 200;
const DEFAULT_LANGUAGE: &str = "en";
const DEFAULT_TRANSLATE: bool = false;
const DEFAULT_CONTEXT: bool = true;

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

struct Settings {
  vad_mode: String,
  min_voice_activity_ms: u64,
  language: String,
  translate: bool,
  context: bool,
}

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
  #[allow(dead_code)]
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
    params.set_suppress_blank(true);
    params.set_suppress_non_speech_tokens(true);
    {
      let settings = self.settings.lock().unwrap();
      match settings.language.as_str() {
        "en" => params.set_language(Some("en")),
        "auto" => params.set_language(Some("auto")),
        other => panic!("unsupported language: {}", other),
      }
      params.set_translate(settings.translate);
      params.set_no_context(!settings.context);
    }
    params
  }

  fn run_model(&self, state: &mut State, chunk: Chunk) -> Result<Option<Buffer>, FlowError> {
    let samples = convert_integer_to_float_audio(&chunk.buffer);

    let start = Instant::now();
    state
      .whisper_state
      .full(self.whisper_params(), &samples)
      .unwrap();
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

        gstreamer::info!(CAT, "{}", segment);

        let segment = format!("{}\n", segment);
        let mut buffer = Buffer::with_size(segment.len()).map_err(|_| FlowError::Error)?;
        let buffer_mut = buffer.get_mut().ok_or(FlowError::Error)?;
        buffer_mut.set_pts(
          chunk
            .start_pts
            .checked_add(ClockTime::from_mseconds(start_ts as u64 * 10))
            .unwrap(),
        );
        buffer_mut.set_duration(ClockTime::from_mseconds(
          (end_ts as u64 - start_ts as u64) * 10,
        ));
        buffer_mut
          .copy_from_slice(0, segment.as_bytes())
          .map_err(|_| FlowError::Error)?;

        Ok(Some(buffer))
      }
      else {
        Ok(None)
      }
    }
    else {
      Ok(None)
    }
  }
}

#[glib::object_subclass]
impl ObjectSubclass for WhisperFilter {
  type ParentType = BaseTransform;
  type Type = super::WhisperFilter;

  const NAME: &'static str = "GstWhisperFilter";

  fn new() -> Self {
    Self {
      settings: Mutex::new(Settings {
        vad_mode: DEFAULT_VAD_MODE.into(),
        min_voice_activity_ms: DEFAULT_MIN_VOICE_ACTIVITY_MS,
        language: DEFAULT_LANGUAGE.into(),
        translate: DEFAULT_TRANSLATE,
        context: DEFAULT_CONTEXT,
      }),
      state: Mutex::new(None),
    }
  }
}

impl ObjectImpl for WhisperFilter {
  fn properties() -> &'static [ParamSpec] {
    static PROPERTIES: Lazy<Vec<ParamSpec>> = Lazy::new(|| {
      vec![
      glib::ParamSpecString::builder("vad-mode")
        .nick("VAD mode")
        .blurb(&format!("The aggressiveness of voice detection. Defaults to '{}'. Other options are 'low-bitrate', 'aggressive' and 'very-aggressive'.", DEFAULT_VAD_MODE))
        .mutable_ready()
        .mutable_paused()
        .mutable_playing()
        .build(),
      glib::ParamSpecInt::builder("min-voice-activity-ms")
        .nick("Minimum voice activity")
        .blurb(&format!("The minimum duration of voice that must be detected for the model to run, in milliseconds. Defaults to {}ms.", DEFAULT_MIN_VOICE_ACTIVITY_MS))
        .mutable_ready()
        .mutable_paused()
        .mutable_playing()
        .build(),
      glib::ParamSpecString::builder("language")
        .nick("Language")
        .blurb(&format!("The target language. Defaults to '{}'. Specify 'auto' to use language detection.", DEFAULT_LANGUAGE))
        .mutable_ready()
        .mutable_paused()
        .mutable_playing()
        .build(),
      glib::ParamSpecBoolean::builder("translate")
        .nick("Translate")
        .blurb(&format!("Whether to translate into the target language. Defaults to {}.", DEFAULT_TRANSLATE))
        .mutable_ready()
        .mutable_paused()
        .mutable_playing()
        .build(),
      glib::ParamSpecBoolean::builder("context")
        .nick("Context")
        .blurb(&format!("Whether to use previous tokens as context for the model. Defaults to {}.", DEFAULT_CONTEXT))
        .mutable_ready()
        .mutable_paused()
        .mutable_playing()
        .build(),
    ]
    });
    PROPERTIES.as_ref()
  }

  fn set_property(&self, _id: usize, value: &Value, pspec: &ParamSpec) {
    let mut settings = self.settings.lock().unwrap();
    match pspec.name() {
      "vad-mode" => {
        settings.vad_mode = value.get().unwrap();
      },
      "min-voice-activity-ms" => {
        settings.min_voice_activity_ms = value.get().unwrap();
      },
      "language" => {
        settings.language = value.get().unwrap();
      },
      "translate" => {
        settings.translate = value.get().unwrap();
      },
      "context" => {
        settings.context = value.get().unwrap();
      },
      other => panic!("no such property: {}", other),
    }
  }

  fn property(&self, _id: usize, pspec: &ParamSpec) -> Value {
    let settings = self.settings.lock().unwrap();
    match pspec.name() {
      "vad-mode" => settings.vad_mode.to_value(),
      "min-voice-activity-ms" => settings.min_voice_activity_ms.to_value(),
      "language" => settings.language.to_value(),
      "translate" => settings.translate.to_value(),
      "context" => settings.context.to_value(),
      other => panic!("no such property: {}", other),
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
      let vad_mode = match self.settings.lock().unwrap().vad_mode.as_str() {
        "quality" => VadMode::Quality,
        "low-bitrate" => VadMode::LowBitrate,
        "aggressive" => VadMode::Aggressive,
        "very-aggressive" => VadMode::VeryAggressive,
        other => panic!("invalid VAD mode: {}", other),
      };
      thread::spawn(move || {
        gstreamer::debug!(CAT, "vad starting");
        let mut vad =
          Vad::new_with_rate_and_mode((SAMPLE_RATE as i32).try_into().unwrap(), vad_mode);
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
    let _ = self.state.lock().unwrap().take();
    gstreamer::debug!(CAT, "stop(): stopped");
    Ok(())
  }

  fn transform_caps(
    &self,
    direction: PadDirection,
    _caps: &Caps,
    maybe_filter: Option<&Caps>,
  ) -> Option<Caps> {
    let mut caps = if direction == PadDirection::Src {
      SINK_CAPS.clone()
    }
    else {
      SRC_CAPS.clone()
    };
    if let Some(filter) = maybe_filter {
      caps = filter.intersect_with_mode(&caps, CapsIntersectMode::First);
    }
    Some(caps)
  }

  fn generate_output(&self) -> Result<GenerateOutputSuccess, FlowError> {
    if let Some(buffer) = self.take_queued_buffer() {
      let mut locked_state = self.state.lock().unwrap();
      let state = locked_state.as_mut().ok_or_else(|| {
        element_imp_error!(
          self,
          CoreError::Negotiation,
          ["Can not generate an output without state"]
        );
        FlowError::NotNegotiated
      })?;

      let buffer_reader = buffer
        .as_ref()
        .map_readable()
        .map_err(|_| FlowError::Error)?;
      let samples = buffer_reader.as_slice_of().map_err(|_| FlowError::Error)?;
      gstreamer::debug!(CAT, "generate_output(): reading {} samples", samples.len());

      let buffer_len = samples.len();
      if buffer_len >= 160 {
        let vad_buffer = if buffer_len >= 480 {
          &samples[buffer_len - 480..buffer_len]
        }
        else if buffer_len >= 320 {
          &samples[buffer_len - 320..buffer_len]
        }
        else {
          &samples[buffer_len - 160..buffer_len]
        };
        state.vad_sender.send(vad_buffer.to_vec()).unwrap();
      }

      if state.voice_activity_detected.load(Ordering::Relaxed) {
        if let Some(chunk) = state.chunk.as_mut() {
          chunk.buffer.extend_from_slice(samples);
        }
        else {
          gstreamer::debug!(CAT, "generate_output(): voice activity started");
          state.chunk = Some(Chunk {
            start_pts: buffer.pts().unwrap(),
            buffer: state
              .prev_buffer
              .drain(..)
              .chain(samples.iter().copied())
              .collect(),
          });
        }

        Ok(GenerateOutputSuccess::NoOutput)
      }
      else {
        state.prev_buffer = samples.to_vec();

        if let Some(chunk) = state.chunk.take() {
          gstreamer::debug!(CAT, "generate_output(): voice activity ended");
          let min_voice_activity_ms = self.settings.lock().unwrap().min_voice_activity_ms;
          if (buffer.pts().unwrap() - chunk.start_pts).mseconds() >= min_voice_activity_ms {
            let maybe_buffer = self.run_model(state, chunk)?;
            Ok(
              maybe_buffer
                .map(GenerateOutputSuccess::Buffer)
                .unwrap_or(GenerateOutputSuccess::NoOutput),
            )
          }
          else {
            gstreamer::debug!(
              CAT,
              "generate_output(): discarding voice activity < {}ms",
              min_voice_activity_ms
            );
            Ok(GenerateOutputSuccess::NoOutput)
          }
        }
        else {
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
