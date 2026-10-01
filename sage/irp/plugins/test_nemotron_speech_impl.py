import tempfile
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

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


class FakeNVCFStub:
    def __init__(self, responses, fail_with=None):
        self.responses = responses
        self.fail_with = fail_with
        self.received_requests = None
        self.metadata = None
        self.timeout = None

    def StreamingRecognize(self, requests, *, metadata, timeout):
        self.received_requests = list(requests)
        self.metadata = metadata
        self.timeout = timeout
        if self.fail_with is not None:
            raise self.fail_with
        return iter(self.responses)


class FakeNVCFClient:
    class AudioEncoding:
        LINEAR_PCM = "LINEAR_PCM"

    def __init__(self, responses, fail_with=None):
        self.responses = responses
        self.fail_with = fail_with
        self.auth = None
        self.service = None

    def Auth(self, **kwargs):
        class Auth:
            def __init__(self):
                self.kwargs = kwargs

            def get_auth_metadata(self):
                return tuple(tuple(pair) for pair in kwargs["metadata_args"])

        self.auth = Auth()
        return self.auth

    def ASRService(self, auth):
        service = SimpleNamespace(stub=FakeNVCFStub(self.responses, self.fail_with))
        self.service = service
        return service

    @staticmethod
    def RecognitionConfig(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def StreamingRecognitionConfig(**kwargs):
        return SimpleNamespace(**kwargs)


class FakeRivaASR:
    @staticmethod
    def streaming_request_generator(chunks, streaming_config):
        yield SimpleNamespace(streaming_config=streaming_config, audio_content=b"")
        for chunk in chunks:
            yield SimpleNamespace(streaming_config=None, audio_content=chunk)


class NemotronSpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def write_wav(self, path, *, seconds=1, channels=1, sample_width=2, rate=16000):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(sample_width)
            output.setframerate(rate)
            output.writeframes(b"\x00" * rate * seconds * channels * sample_width)
        return path

    def test_offline_nim_transcription_is_untrusted_and_uses_documented_endpoint(self):
        audio = self.root / "sample.wav"
        audio.write_bytes(b"sample-audio")
        Handler.response_body = b'{"text":"recognized words"}'
        Handler.received_authorization = None

        plugin = NemotronSpeechASR({"base_url": self.base_url, "api_key": "test-token"})
        final = plugin.step(plugin.init_state(audio, {}))

        self.assertEqual(final.x["transcript"], "recognized words")
        self.assertEqual(Handler.received_authorization, "Bearer test-token")
        self.assertEqual(plugin.extract(final), {
            "text": "recognized words",
            "confidence": None,
            "source": "nvidia_nemotron_speech_asr_nim",
            "trusted": False,
            "authority": "none",
        })

    def test_offline_nim_bad_response_fails_closed(self):
        audio = self.root / "sample.flac"
        audio.write_bytes(b"sample-audio")
        Handler.response_body = b'{"confidence":0.99}'
        plugin = NemotronSpeechASR({"base_url": self.base_url})
        with self.assertRaisesRegex(NemotronSpeechError, "missing transcript text"):
            plugin.step(plugin.init_state(audio, {}))

    def test_unsafe_offline_endpoint_configuration_is_rejected(self):
        for url in (
            "http://speech.example.com",
            "https://user:password@speech.example.com",
            "https://speech.example.com/custom-path",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                NemotronSpeechASR({"base_url": url})

    def test_audio_size_and_suffix_are_checked(self):
        plugin = NemotronSpeechASR({"base_url": self.base_url})
        wrong_type = self.root / "sample.mp3"
        wrong_type.write_bytes(b"audio")
        with self.assertRaisesRegex(ValueError, "WAV, OPUS, or FLAC"):
            plugin.init_state(wrong_type, {})

        too_large = self.root / "large.wav"
        too_large.write_bytes(b"0" * (plugin.MAX_AUDIO_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "25 MiB"):
            plugin.init_state(too_large, {})

    def test_nvcf_nemotron_streaming_uses_riva_auth_deadline_and_final_results(self):
        audio = self.write_wav(self.root / "nvcf.wav", seconds=2)
        final_response = SimpleNamespace(results=[
            SimpleNamespace(is_final=False, alternatives=[SimpleNamespace(transcript="partial")]),
            SimpleNamespace(is_final=True, alternatives=[SimpleNamespace(transcript="recognized ")]),
            SimpleNamespace(is_final=True, alternatives=[SimpleNamespace(transcript="words")]),
        ])
        fake = FakeNVCFClient([final_response])
        plugin = NemotronSpeechASR(
            {
                "api_key": "test-token",
                "function_id": "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa",
                "timeout_seconds": 12,
            },
            nvcf_client_factory=lambda: (fake, FakeRivaASR),
        )

        final = plugin.step(plugin.init_state(audio, {}))

        self.assertEqual(final.x["transcript"], "recognized words")
        self.assertEqual(fake.auth.kwargs["uri"], "grpc.nvcf.nvidia.com:443")
        self.assertEqual(fake.auth.kwargs["use_ssl"], True)
        self.assertIn(["function-id", "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"], fake.auth.kwargs["metadata_args"])
        self.assertIn(["authorization", "Bearer test-token"], fake.auth.kwargs["metadata_args"])
        self.assertEqual(fake.service.stub.timeout, 12.0)
        self.assertEqual(len(fake.service.stub.received_requests), 3)
        self.assertEqual(fake.service.stub.received_requests[0].streaming_config.config.language_code, "en-US")
        self.assertTrue(all(len(req.audio_content) == 32000 for req in fake.service.stub.received_requests[1:]))
        self.assertFalse(plugin.extract(final)["trusted"])
        self.assertEqual(plugin.extract(final)["authority"], "none")

    def test_nvcf_rejects_missing_credentials_and_non_pcm_input(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "", "NVIDIA_NEMOTRON_ASR_FUNCTION_ID": ""}):
            with self.assertRaisesRegex(ValueError, "NVIDIA_API_KEY"):
                NemotronSpeechASR({"function_id": "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"})
        plugin = NemotronSpeechASR({
            "api_key": "test-token",
            "function_id": "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa",
        })
        stereo = self.write_wav(self.root / "stereo.wav", channels=2)
        with self.assertRaisesRegex(ValueError, "mono, 16-bit PCM"):
            plugin.init_state(stereo, {})

    def test_nvcf_transport_failure_is_sanitized_and_fails_closed(self):
        audio = self.write_wav(self.root / "failure.wav")
        fake = FakeNVCFClient([], fail_with=RuntimeError("Bearer secret-token"))
        plugin = NemotronSpeechASR(
            {
                "api_key": "test-token",
                "function_id": "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa",
            },
            nvcf_client_factory=lambda: (fake, FakeRivaASR),
        )
        with self.assertRaisesRegex(NemotronSpeechError, "streaming request failed") as raised:
            plugin.step(plugin.init_state(audio, {}))
        self.assertNotIn("secret-token", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
