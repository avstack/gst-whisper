mod imp;

use gstreamer::{glib, prelude::StaticType, Rank};

glib::wrapper! {
  pub struct WhisperFilter(ObjectSubclass<imp::WhisperFilter>) @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
}

pub fn register(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
  gstreamer::Element::register(
    Some(plugin),
    "whisper",
    Rank::None,
    WhisperFilter::static_type(),
  )
}
