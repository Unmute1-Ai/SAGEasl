from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from sage.irp.plugins.nemotron_speech_impl import NemotronSpeechASR, NemotronSpeechError


class Handler(BaseHTTPRequestHandler):
    response_body = b'{"text":"recognized words"}'
    received_authorization = None

    def do_POST(self):
        self.__class__.received_authorization = self.headers.get("Authorization")
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.__class__.response_body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def nim_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_transcription_is_untrusted_observation_and_uses_nim_endpoint(tmp_path, nim_server):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"sample-audio")
    Handler.response_body = b'{"text":"recognized words"}'
    Handler.received_authorization = None

    plugin = NemotronSpeechASR({"base_url": nim_server, "api_key": "test-token"})
    initial = plugin.init_state(audio, {})
    final = plugin.step(initial)

    assert final.x["transcript"] == "recognized words"
    assert Handler.received_authorization == "Bearer test-token"
    assert plugin.extract(final) == {
        "text": "recognized words",
        "confidence": None,
        "source": "nvidia_nemotron_speech_asr_nim",
        "trusted": False,
        "authority": "none",
    }


def test_bad_response_fails_closed(tmp_path, nim_server):
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"sample-audio")
    Handler.response_body = b'{"confidence":0.99}'
    plugin = NemotronSpeechASR({"base_url": nim_server})
    with pytest.raises(NemotronSpeechError, match="missing transcript text"):
        plugin.step(plugin.init_state(audio, {}))


@pytest.mark.parametrize("url", [
    "http://speech.example.com",
    "https://user:password@speech.example.com",
    "https://speech.example.com/custom-path",
])
def test_unsafe_endpoint_configuration_is_rejected(url):
    with pytest.raises(ValueError):
        NemotronSpeechASR({"base_url": url})


def test_audio_size_and_suffix_are_checked(tmp_path, nim_server):
    plugin = NemotronSpeechASR({"base_url": nim_server})
    wrong_type = tmp_path / "sample.mp3"
    wrong_type.write_bytes(b"audio")
    with pytest.raises(ValueError, match="WAV, OPUS, or FLAC"):
        plugin.init_state(wrong_type, {})

    too_large = tmp_path / "large.wav"
    too_large.write_bytes(b"0" * (plugin.MAX_AUDIO_BYTES + 1))
    with pytest.raises(ValueError, match="25 MiB"):
        plugin.init_state(too_large, {})

