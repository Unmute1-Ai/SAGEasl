"""SAGE IRP adapter for NVIDIA Nemotron Speech ASR NIM.

Uses NVIDIA's documented offline multipart HTTP endpoint. The returned text
is an untrusted observation for downstream reasoning; this plugin never
authorizes or dispatches effects.
"""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from sage.irp.base import IRPPlugin, IRPState


class NemotronSpeechError(RuntimeError):
    """ASR request could not be completed or its response was invalid."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class NemotronSpeechASR(IRPPlugin):
    """One-shot audio transcription plugin using a preconfigured Speech NIM."""

    SUPPORTED_SUFFIXES = {".wav", ".opus", ".flac"}
    MAX_AUDIO_BYTES = 25 * 1024 * 1024
    MAX_RESPONSE_BYTES = 1024 * 1024

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        base_url = config.get("base_url") or os.environ.get("NEMOTRON_SPEECH_URL")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Nemotron Speech base_url or NEMOTRON_SPEECH_URL is required")
        parsed = urllib.parse.urlsplit(base_url.strip())
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise ValueError("Nemotron Speech URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Nemotron Speech URL must not contain credentials, query, or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("Nemotron Speech base_url must not include an API path")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Plain HTTP is permitted only for loopback development endpoints")

        self.endpoint = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/v1/audio/transcriptions", "", ""))
        self.language = config.get("language", "en-US")
        if not isinstance(self.language, str) or not re.fullmatch(r"[A-Za-z0-9-]{2,35}", self.language):
            raise ValueError("language must be a valid language-code token")
        self.timeout = float(config.get("timeout_seconds", 30.0))
        if not 0 < self.timeout <= 300:
            raise ValueError("timeout_seconds must be in the range (0, 300]")
        self.api_key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY")
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key.strip()):
            raise ValueError("api_key must be a non-empty string when provided")
        self._opener = urllib.request.build_opener(_NoRedirect())

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        audio_path = x0.get("audio_path") if isinstance(x0, dict) else x0
        if not isinstance(audio_path, (str, os.PathLike)):
            raise ValueError("x0 must be an audio file path")
        path = Path(audio_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError("audio_path must be a WAV, OPUS, or FLAC file")
        if path.stat().st_size <= 0 or path.stat().st_size > self.MAX_AUDIO_BYTES:
            raise ValueError("audio file is empty or exceeds the 25 MiB limit")
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

