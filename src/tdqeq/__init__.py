"""
Tdqeq: A powerful PDF table detection and extraction pipeline.
"""

# PyTorch 2.6+ changed the default of `weights_only` to True for torch.load.
# Since doclayout_yolo (YOLOv10) uses torch.load internally and requires loading
# custom Python class instances, we patch torch.load to default to weights_only=False
# when not explicitly specified.
try:
    import torch

    original_load = torch.load

    def safe_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = safe_load
except ImportError:
    pass
