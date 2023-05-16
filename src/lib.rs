use gstreamer::glib;

mod filter;

fn plugin_init(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
  filter::register(plugin)?;
  Ok(())
}

gstreamer::plugin_define!(
  whisper,
  env!("CARGO_PKG_DESCRIPTION"),
  plugin_init,
  concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
  "MIT/Apache-2.0",
  env!("CARGO_PKG_NAME"),
  env!("CARGO_PKG_NAME"),
  env!("CARGO_PKG_REPOSITORY"),
  env!("BUILD_REL_DATE")
);
