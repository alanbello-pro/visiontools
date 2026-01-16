from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Detection:
    track_id: int
    box: np.ndarray
    class_name: str
    confidence: float


@dataclass
class FrameTrackData:
    frame_id: int
    frame_anotado: np.ndarray
    registros_csv: List[dict]
    fps: float
