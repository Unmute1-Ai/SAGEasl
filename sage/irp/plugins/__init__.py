"""
IRP Plugin implementations for different modalities.
"""

# Heavy dependencies (torch, etc.) — optional for minimal envs (Sprout, arc venv)
try:
    from .vision import VisionIRP
    from .language import LanguageIRP
    from .control import ControlIRP
    from .memory import MemoryIRP
    from .tinyvae_irp_plugin import TinyVAEIRP, create_tinyvae_irp
    _FULL = True
except ImportError:
    _FULL = False

__all__ = []
if _FULL:
    __all__ += ['VisionIRP', 'LanguageIRP', 'ControlIRP', 'MemoryIRP',
                'TinyVAEIRP', 'create_tinyvae_irp']

# Speech NIM adapter is lightweight and does not require torch, riva-client,
# or model downloads. Keep it independently importable in minimal deployments.
try:
    from .nemotron_speech_impl import NemotronSpeechASR
    __all__.append('NemotronSpeechASR')
except ImportError:
    pass

