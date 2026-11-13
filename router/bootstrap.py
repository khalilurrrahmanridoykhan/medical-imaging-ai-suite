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

    try:
        from modalities.ct.infer import predict as ct_predict

        register("ct", ct_predict)
    except ImportError:
        pass

    try:
        from modalities.mri.infer import predict as mri_predict

        register("mri", mri_predict)
    except ImportError:
        pass

    try:
        from modalities.ultrasound.infer import predict as ultrasound_predict

        register("ultrasound", ultrasound_predict)
    except ImportError:
        pass

    try:
        from modalities.ecg.infer import predict as ecg_predict

        register("ecg", ecg_predict)
    except ImportError:
        pass
