import numpy as np

from src.models.data_models import Detection
from src.utils.geometry import scale_bounding_box

def scale_detections(detections: list[Detection], scale_factor: float) -> list[Detection]:
    if scale_factor == 1.0:
        return detections
    
    scaled_detections = []
    for det in detections:
        scaled_box = scale_bounding_box(det.box, scale_factor)
        scaled_det = Detection(
            track_id=det.track_id,
            box=scaled_box,
            confidence=det.confidence,
            class_name=det.class_name
        )
        scaled_detections.append(scaled_det)
    
    return scaled_detections


def prepare_tracking_data(
    scaled_detections: list[Detection], 
    frame: np.ndarray
) -> tuple[dict[int, dict], dict[int, np.ndarray]]:

    current_tracks: dict[int, dict] = {}
    histogram_map: dict[int, np.ndarray] = {}

    for det in scaled_detections:
        original_id: int = det.track_id
        box: np.ndarray = det.box
        classe_nome: str = det.class_name

        current_tracks[original_id] = {
            'box': box,
            'classe': classe_nome,
            'histogram': None 
        }
        
    return current_tracks, histogram_map