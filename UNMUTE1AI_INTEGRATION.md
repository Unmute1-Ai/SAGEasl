# Unmute1AI Integration Notes

This repository contains SAGE research/software with its own upstream authorship and AGPL licensing metadata.

For Unmute1AI, the useful integration boundary is:

```text
SAGE cognition / candidate planning
        |
        v
U1 Sentinel external authority gate
        |
        v
bounded / simulated effect adapters
```

SAGE may propose or rank actions. It should not inherit payment, health, emergency, smart-home, or physical authority from model capability.

Unmute1AI-specific changes should preserve upstream attribution and be isolated/documented rather than rewriting project history.

## NVIDIA Nemotron Speech ASR

`sage.irp.plugins.NemotronSpeechASR` is an optional one-shot SAGE IRP plugin.
Its default path streams a local PCM WAV file to NVIDIA's hosted Nemotron ASR
Streaming function through the documented Riva/NVCF gRPC API. Install the
optional client dependency with `python -m pip install -r
nvidia-speech-requirements.txt`. Provide `NVIDIA_API_KEY` through a process
environment or secret manager, and set
`NVIDIA_NEMOTRON_ASR_FUNCTION_ID` to the active function ID from NVIDIA's
Nemotron ASR catalog. Function IDs are deployment-specific and are not
hardcoded. The client uses TLS, the Riva `function-id` and bearer metadata,
and a bounded gRPC deadline.

```python
from sage.irp.plugins import NemotronSpeechASR

asr = NemotronSpeechASR({"language": "en-US", "timeout_seconds": 60})
final_state, history = asr.refine("recording.wav", {})
transcript_candidate = asr.extract(final_state)
```

The hosted Nemotron streaming path accepts local mono, 16-bit PCM WAV files up
to 25 MiB. It does not start a microphone, deploy a model, infer confidence, or
fall back to another recognizer. Its output is explicitly marked
`trusted=False` and `authority=none`; it is an observation for downstream SAGE
reasoning, and any proposed action still passes the external U1 Sentinel gate.
Transport failures, timeouts, malformed responses, and unavailable endpoints
raise `NemotronSpeechError` and produce no transcript candidate.

For an already deployed, offline-capable Speech NIM, explicitly select
`mode="nim_offline"` and provide `base_url` (or `NEMOTRON_SPEECH_URL`). This
uses multipart `POST /v1/audio/transcriptions`; the deployed model profile must
support offline inference. Nemotron ASR Streaming itself is streaming-only, so
it must use the default Riva/NVCF path above, not the offline HTTP route. Remote
HTTP endpoints require HTTPS; plain HTTP is accepted only for loopback
development. If that private endpoint requires authentication, use its own
`NEMOTRON_SPEECH_API_KEY` secret or pass `api_key` in the application's
secret-backed config. The hosted `NVIDIA_API_KEY` is never forwarded to this
separate HTTP endpoint. Keep all credentials out of source control.
