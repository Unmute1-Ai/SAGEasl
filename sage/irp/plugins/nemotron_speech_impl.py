"""SAGE IRP adapter for NVIDIA Nemotron Speech ASR NIM.

Uses NVIDIA's documented offline multipart HTTP endpoint. The returned text
is an untrusted observation for downstream reasoning; this plugin never
authorizes or dispatches effects.
"""

from __future__ import annotations

import json
import io
import mimetypes
import os
from pathlib import Path
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from typing import Any, Dict, List

from sage.irp.base import IRPPlugin, IRPState


class NemotronSpeechError(RuntimeError):
    """ASR request could not be completed or its response was invalid."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class NemotronSpeechASR(IRPPlugin):
    """One-shot SAGE observation adapter for NVIDIA speech inference.

    The default path uses NVIDIA's hosted Nemotron ASR Streaming Riva API.
    ``mode="nim_offline"`` selects the separate multipart endpoint exposed by
    self-hosted offline-capable Speech NIM models.
    """

    SUPPORTED_SUFFIXES = {".wav", ".opus", ".flac"}
    MAX_AUDIO_BYTES = 25 * 1024 * 1024
    MAX_RESPONSE_BYTES = 1024 * 1024

    def __init__(self, config: Dict[str, Any], *, nvcf_client_factory=None):
        super().__init__(config)
        base_url = config.get("base_url") or os.environ.get("NEMOTRON_SPEECH_URL")
        self.mode = config.get("mode", "nim_offline" if base_url else "nvcf_streaming")
        if self.mode not in {"nim_offline", "nvcf_streaming"}:
            raise ValueError("mode must be 'nvcf_streaming' or 'nim_offline'")

        self.endpoint = None
        self.function_id = None
        self.grpc_endpoint = None
        configured_api_key = config.get("api_key")
        self._nvcf_client_factory = nvcf_client_factory
        if self.mode == "nim_offline":
            self.api_key = configured_api_key or os.environ.get("NEMOTRON_SPEECH_API_KEY")
            if self.api_key is not None and (
                not isinstance(self.api_key, str) or not self.api_key.strip()
            ):
                raise ValueError("api_key must be a non-empty string when provided")
            self.endpoint = self._validate_nim_base_url(base_url)
        else:
            self.api_key = configured_api_key or os.environ.get("NVIDIA_API_KEY")
            if not isinstance(self.api_key, str) or not self.api_key.strip():
                raise ValueError("NVCF mode requires NVIDIA_API_KEY (or api_key from a secret manager)")
            self.api_key = self.api_key.strip()
            self.function_id = config.get("function_id") or os.environ.get("NVIDIA_NEMOTRON_ASR_FUNCTION_ID")
            try:
                parsed_function_id = uuid.UUID(str(self.function_id))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(
                    "NVCF mode requires the active NVIDIA Nemotron function ID in "
                    "NVIDIA_NEMOTRON_ASR_FUNCTION_ID or function_id"
                ) from exc
            self.function_id = str(parsed_function_id)
            self.grpc_endpoint = "grpc.nvcf.nvidia.com:443"

        self.language = config.get("language", "en-US")
        if not isinstance(self.language, str) or not re.fullmatch(r"[A-Za-z0-9-]{2,35}", self.language):
            raise ValueError("language must be a valid language-code token")
        self.timeout = float(config.get("timeout_seconds", 30.0))
        if not 0 < self.timeout <= 300:
            raise ValueError("timeout_seconds must be in the range (0, 300]")
        if self.mode == "nim_offline":
            self._opener = urllib.request.build_opener(_NoRedirect())

    @staticmethod
    def _validate_nim_base_url(base_url):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("nim_offline mode requires base_url or NEMOTRON_SPEECH_URL")
        parsed = urllib.parse.urlsplit(base_url.strip())
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("Nemotron Speech URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Nemotron Speech URL must not contain credentials, query, or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("Nemotron Speech base_url must not include an API path")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Plain HTTP is permitted only for loopback development endpoints")
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/v1/audio/transcriptions", "", ""))

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        audio_path = x0.get("audio_path") if isinstance(x0, dict) else x0
        if not isinstance(audio_path, (str, os.PathLike)):
            raise ValueError("x0 must be an audio file path")
        path = Path(audio_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("audio_path must name a file")
        supported = {".wav"} if self.mode == "nvcf_streaming" else self.SUPPORTED_SUFFIXES
        if path.suffix.lower() not in supported:
            if self.mode == "nvcf_streaming":
                raise ValueError("NVCF Nemotron streaming requires a mono, 16-bit PCM WAV file")
            raise ValueError("audio_path must be a WAV, OPUS, or FLAC file")
        if path.stat().st_size <= 0 or path.stat().st_size > self.MAX_AUDIO_BYTES:
            raise ValueError("audio file is empty or exceeds the 25 MiB limit")
        if self.mode == "nvcf_streaming":
            try:
                with wave.open(str(path), "rb") as wav:
                    if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                        raise ValueError("NVCF Nemotron streaming requires a mono, 16-bit PCM WAV file")
                    if wav.getframerate() < 8000 or wav.getframerate() > 96000:
                        raise ValueError("WAV sample rate must be between 8000 and 96000 Hz")
            except (wave.Error, EOFError) as exc:
                raise ValueError("NVCF Nemotron streaming requires a valid PCM WAV file") from exc
        return IRPState(
            x={"audio_path": str(path), "transcript": None},
            meta={"untrusted_observation": True},
        )

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        if state.x.get("transcript") is not None:
            return state
        path = Path(state.x["audio_path"])
        try:
            audio = path.read_bytes()
            if not audio or len(audio) > self.MAX_AUDIO_BYTES:
                raise NemotronSpeechError("audio file changed or exceeds the size limit")
            transcript = self._transcribe(path, audio)
        except NemotronSpeechError:
            raise
        except Exception as exc:
            # Hide endpoint, token, and remote response details from ordinary
            # SAGE telemetry and callers; the operation remains failed closed.
            raise NemotronSpeechError("Nemotron Speech transcription failed") from exc
        return IRPState(
            x={"audio_path": str(path), "transcript": transcript},
            step_idx=state.step_idx + 1,
            energy_val=0.0,
            meta={**state.meta, "untrusted_observation": True, "asr_completed": True},
        )

    def _transcribe(self, path: Path, audio: bytes) -> str:
        if self.mode == "nvcf_streaming":
            return self._transcribe_nvcf(path, audio)
        return self._transcribe_nim_offline(path, audio)

    def _transcribe_nim_offline(self, path: Path, audio: bytes) -> str:
        boundary = "u1" + secrets.token_hex(16)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        fields = [("language", self.language)]
        parts = []
        for name, value in fields:
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n".encode("utf-8")
            )
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio{path.suffix.lower()}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8") + audio + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode("ascii"))
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=b"".join(parts), headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise NemotronSpeechError(f"Nemotron Speech returned HTTP {response.status}")
                body = response.read(self.MAX_RESPONSE_BYTES + 1)
                if len(body) > self.MAX_RESPONSE_BYTES:
                    raise NemotronSpeechError("Nemotron Speech response exceeded the size limit")
        except urllib.error.HTTPError as exc:
            raise NemotronSpeechError(f"Nemotron Speech returned HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise NemotronSpeechError("Nemotron Speech endpoint is unavailable") from None

        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NemotronSpeechError("Nemotron Speech returned invalid JSON") from exc
        if not isinstance(decoded, dict) or not isinstance(decoded.get("text"), str):
            raise NemotronSpeechError("Nemotron Speech response is missing transcript text")
        return decoded["text"]

    def _transcribe_nvcf(self, path: Path, audio: bytes) -> str:
        try:
            if self._nvcf_client_factory is None:
                import riva.client
                import riva.client.asr
                riva_client = riva.client
                riva_asr = riva.client.asr
            else:
                riva_client, riva_asr = self._nvcf_client_factory()

            with wave.open(io.BytesIO(audio), "rb") as wav:
                sample_rate = wav.getframerate()
                pcm = wav.readframes(wav.getnframes())
            # Keep the entire audio inside the configured request deadline; use
            # one-second chunks so the streaming-only Nemotron endpoint receives
            # the documented Riva streaming request shape.
            chunk_bytes = sample_rate * 2
            chunks = (pcm[offset:offset + chunk_bytes] for offset in range(0, len(pcm), chunk_bytes))
            auth = riva_client.Auth(
                use_ssl=True,
                uri=self.grpc_endpoint,
                metadata_args=[
                    ["function-id", self.function_id],
                    ["authorization", f"Bearer {self.api_key}"],
                ],
            )
            service = riva_client.ASRService(auth)
            recognition = riva_client.RecognitionConfig(
                language_code=self.language,
                encoding=riva_client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=sample_rate,
                audio_channel_count=1,
                max_alternatives=1,
                enable_automatic_punctuation=True,
            )
            streaming = riva_client.StreamingRecognitionConfig(config=recognition, interim_results=False)
            requests = riva_asr.streaming_request_generator(chunks, streaming)
            responses = service.stub.StreamingRecognize(
                requests,
                metadata=auth.get_auth_metadata(),
                timeout=self.timeout,
            )
            final_segments = []
            for response in responses:
                for result in getattr(response, "results", ()):
                    if not getattr(result, "is_final", False):
                        continue
                    alternatives = getattr(result, "alternatives", ())
                    if alternatives and isinstance(alternatives[0].transcript, str):
                        final_segments.append(alternatives[0].transcript)
            return "".join(final_segments)
        except NemotronSpeechError:
            raise
        except Exception as exc:
            # Do not leak NVCF auth metadata, endpoint details, or provider
            # response data through the SAGE telemetry/error surface.
            raise NemotronSpeechError("Nemotron ASR streaming request failed") from exc

    def energy(self, state: IRPState) -> float:
        # Completion status only. ASR confidence is deliberately not inferred.
        return 0.0 if isinstance(state.x, dict) and isinstance(state.x.get("transcript"), str) else 1.0

    def halt(self, history: List[IRPState]) -> bool:
        return bool(history and self.energy(history[-1]) == 0.0)

    def extract(self, state: IRPState) -> Dict[str, Any]:
        if not isinstance(state.x, dict) or not isinstance(state.x.get("transcript"), str):
            raise NemotronSpeechError("no completed transcription is available")
        return {
            "text": state.x["transcript"],
            "confidence": None,
            "source": "nvidia_nemotron_speech_asr_nim",
            "trusted": False,
            "authority": "none",
        }
