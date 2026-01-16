import sys
from ultralytics import solutions
import torch

from src.setup.config import GeometryConfig, ModelConfig

def extract_current_tracks(trackzone) -> dict:

    current_tracks = {}
    if hasattr(trackzone, 'track_ids') and trackzone.track_ids is not None:
        for i, track_id in enumerate(trackzone.track_ids):
            current_tracks[int(track_id)] = trackzone.boxes[i]
    return current_tracks

def determine_yolo_device() -> str:
    if sys.platform == 'darwin': # macOS
        if torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif sys.platform.startswith('win') or sys.platform.startswith('linux'): # Windows ou Linux
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        return "cpu"

def initialize_trackzone(model, geometry_config: GeometryConfig, model_config: ModelConfig):
    return solutions.TrackZone(
        model=model,
        region=geometry_config.REGIAO_1,
        show=False,
        conf=model_config.TRACKZONE_CONF,
        iou=model_config.TRACKZONE_IOU,
        classes=model_config.TRACKZONE_CLASSES,
        device=model_config.TRACKZONE_DEVICE
    )