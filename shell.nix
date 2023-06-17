with import <nixpkgs> {};
mkShell {
  name = "gst-whisper";
  buildInputs = [
    cargo
    cargo-c
    cmake
    pkg-config
    git
    glib
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-good
    gst_all_1.gst-plugins-bad
  ] ++ (if stdenv.isDarwin then [
    darwin.apple_sdk.frameworks.Accelerate
  ] else []);

  shellHook = ''
    export HISTFILE="$HOME/.local/log/bash_history/gst-whisper";
  '';
}
