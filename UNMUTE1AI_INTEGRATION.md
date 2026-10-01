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

`sage.irp.plugins.NemotronSpeechASR` is an optional one-shot SAGE IRP plugin for
an already deployed NVIDIA Nemotron Speech ASR NIM. It uses the documented
offline `POST /v1/audio/transcriptions` multipart endpoint (default HTTP port
9000). Configure `base_url` explicitly or set `NEMOTRON_SPEECH_URL`; remote
endpoints must use HTTPS. Plain HTTP is accepted only for loopback development.
For an authorized NVIDIA-hosted endpoint, provide `NVIDIA_API_KEY` through the
process environment or a secret manager; never store it in source control.

```python
from sage.irp.plugins import NemotronSpeechASR

asr = NemotronSpeechASR({"base_url": "https://speech.example.internal:9000"})
final_state, history = asr.refine("recording.wav", {})
transcript_candidate = asr.extract(final_state)
```

Accepted inputs are local WAV, OPUS, and FLAC files up to 25 MiB. The plugin
does not start a microphone, deploy a model, infer confidence, or fall back to
another recognizer. Its output is explicitly marked `trusted=False` and
`authority=none`; it is an observation for downstream SAGE reasoning, and any
proposed action still passes the external U1 Sentinel gate. HTTP failures,
redirects, malformed responses, and unavailable endpoints raise
`NemotronSpeechError` and produce no transcript candidate.

