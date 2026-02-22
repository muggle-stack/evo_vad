"""
EvoVAD - Voice Activity Detection SDK

Usage:
    import evo_vad

    # Quick detection
    result = evo_vad.detect(audio_array)
    print(f"Speech probability: {result.probability}")

    # Using engine
    engine = evo_vad.VadEngine()
    result = engine.detect(audio_array)
    if result.is_speech:
        print("Speech detected!")

    # Streaming mode
    callback = evo_vad.VadCallback()
    callback.on_speech_start(lambda ts: print(f"Speech started at {ts}ms"))
    callback.on_speech_end(lambda ts, dur: print(f"Speech ended, duration {dur}ms"))

    engine.set_callback(callback)
    engine.start()
    for frame in audio_frames:
        engine.send_audio_frame(frame)
    engine.stop()
"""

from ._evo_vad import (
    # Enums
    VadState,
    VadBackendType,
    # Config
    VadConfig,
    # Result
    VadResult,
    # Engine
    VadEngine,
    # Callback
    VadCallback,
    # Quick function
    detect,
    # Module info
    __version__,
)

__all__ = [
    # Enums
    "VadState",
    "VadBackendType",
    # Config
    "VadConfig",
    # Result
    "VadResult",
    # Engine
    "VadEngine",
    # Callback
    "VadCallback",
    # Quick function
    "detect",
    # Module info
    "__version__",
]
