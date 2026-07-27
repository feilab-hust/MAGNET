"""Compatibility import for the tile-prediction entry point.

Training and tiled inference now share the same pseudo-SR dataloader. Keeping
this module avoids breaking old imports without maintaining a second divergent
copy of the data pipeline.
"""
from .dataloader_MultiTask import get_multi_data

__all__ = ["get_multi_data"]
