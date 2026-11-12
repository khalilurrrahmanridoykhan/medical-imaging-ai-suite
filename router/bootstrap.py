"""Import each implemented modality's infer module and register it.

Modalities not built yet (see README roadmap) are skipped silently so the
CLI keeps working as phases land one at a time.
"""

from router.registry import register


def register_all() -> None:
    try:
        from modalities.xray.infer import predict as xray_predict

        register("xray", xray_predict)
    except ImportError:
        pass
