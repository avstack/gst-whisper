# gst-whisper: A GStreamer element that does speech-to-text using Whisper.

Accepts 16kHz S16 audio buffers on its sink pad and produces text buffers on its source pad.

## Installation

gst-whisper is written in Rust and uses the cargo-c helper. Set up a Rust development environment (e.g. using [rustup](https://rustup.rs)) and then:

```
cargo install cargo-c

git clone https://github.com/avstack/gst-whisper
cd gst-whisper
cargo cbuild --release
export GST_PLUGIN_PATH=$(pwd)/target/release
```

## Example usage

You must already have the Whisper model.

```
export WHISPER_MODEL_PATH=../whisper.cpp/models/ggml-base.en.bin

gst-launch-1.0 --no-position autoaudiosrc ! audioconvert ! audioresample ! queue ! whisper ! fdsink
```

## License

gst-whisper is licensed under either of

* Apache License, Version 2.0, (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Any kinds of contributions are welcome as a pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in these crates by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Acknowledgements

gst-meet development is sponsored by [AVStack](https://avstack.io/). We provide globally-distributed, scalable, managed Jitsi Meet backends.
