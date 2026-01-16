import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pipelines.feature_extractor import FeatureExtractor
from src.utils.geometry import calcular_iou
from src.models.data_models import Detection

class TrackingManager:
    def __init__(self, max_frames_lost: int, iou_threshold: float, color_weight: float):
        self.id_map = {}
        self.track_class_map = {}
        self.track_histogram_map = {}
        self.lost_tracks = {}
        self.previous_tracks = {}

        self.max_frames_lost = max_frames_lost
        self.iou_threshold = iou_threshold
        self.color_weight = color_weight

    def update_tracks(self, current_tracks_data: dict, frame_count: int):

        for track_id, data in current_tracks_data.items():
            if data.get('histogram') is not None:
                self.track_histogram_map[track_id] = data['histogram']

        current_tracks = {tid: data['box'] for tid, data in current_tracks_data.items()}
        current_track_classes = {tid: data['classe'] for tid, data in current_tracks_data.items() if 'classe' in data}

        previous_ids = set(self.previous_tracks.keys())
        current_ids = set(current_tracks.keys())

        self._handle_lost_tracks(previous_ids, current_ids, frame_count)

        new_ids = current_ids - previous_ids

        if new_ids and self.lost_tracks:
            self._reassociate_tracks(new_ids, current_tracks, current_track_classes)

        self._cleanup_old_lost_tracks(frame_count)

        self.previous_tracks = current_tracks

    def _handle_lost_tracks(self, previous_ids: set, current_ids: set, frame_count: int):
        lost_ids = previous_ids - current_ids
        for lost_id in lost_ids:
            if lost_id not in self.lost_tracks:
                final_id = self.get_final_id(lost_id)
                self.lost_tracks[lost_id] = {
                    'box': self.previous_tracks[lost_id],
                    'frame': frame_count,
                    'histogram': self.track_histogram_map.get(lost_id),
                    'classe': self.track_class_map.get(final_id)
                }

    def _reassociate_tracks(self, new_ids: set, current_tracks: dict, current_track_classes: dict):
        new_id_list = list(new_ids)
        new_boxes = [current_tracks[new_id] for new_id in new_id_list]
        new_histograms = [self.track_histogram_map.get(new_id) for new_id in new_id_list]
        new_classes = [current_track_classes.get(new_id) for new_id in new_id_list]

        lost_ids_list = list(self.lost_tracks.keys())
        lost_items = list(self.lost_tracks.values())
        lost_boxes = [item['box'] for item in lost_items]
        lost_histograms = [item['histogram'] for item in lost_items]
        lost_classes = [item['classe'] for item in lost_items]

        valid_pairs = []
        for i, lost_class in enumerate(lost_classes):
            if lost_class is None:
                continue
            for j, new_class in enumerate(new_classes):
                if lost_class == new_class:
                    valid_pairs.append((i, j))
        
        if not valid_pairs:
            return
        
        cost_matrix = np.full((len(lost_boxes), len(new_boxes)), np.inf)
        has_valid_assignment = False

        for i, j in valid_pairs:
            lost_box = lost_boxes[i]
            new_box = new_boxes[j]

            iou = calcular_iou(lost_box, new_box)
            if iou < self.iou_threshold:
                continue

            has_valid_assignment = True
            lost_hist = lost_histograms[i]
            new_hist = new_histograms[j]
            
            iou_cost = 1 - iou
            hist_dist = FeatureExtractor.compare_histograms(lost_hist, new_hist)

            combined_cost = (1 - self.color_weight) * iou_cost + self.color_weight * hist_dist
            cost_matrix[i, j] = combined_cost

        if not has_valid_assignment:
            return

        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ValueError:
            return

        reassociated_ids = set()
        for row, col in zip(row_indices, col_indices):
            if np.isinf(cost_matrix[row, col]):
                continue

            lost_id = lost_ids_list[row]
            new_id = new_id_list[col]
            
            self.id_map[new_id] = self.get_final_id(lost_id)
            
            if new_id in self.track_histogram_map:
                self.track_histogram_map[self.get_final_id(lost_id)] = self.track_histogram_map[new_id]

            reassociated_ids.add(lost_id)

        for lost_id in reassociated_ids:
            if lost_id in self.lost_tracks:
                del self.lost_tracks[lost_id]

    def _cleanup_old_lost_tracks(self, frame_count: int):
        ids_to_remove = [
            lost_id for lost_id, data in self.lost_tracks.items()
            if frame_count - data['frame'] > self.max_frames_lost
        ]
        for lost_id in ids_to_remove:
            self.lost_tracks.pop(lost_id, None)

    def get_final_id(self, original_id: int) -> int:
        return self.id_map.get(original_id, original_id)

    def get_or_set_class(self, track_id: int, class_name: str) -> str:
        if track_id not in self.track_class_map:
            self.track_class_map[track_id] = class_name
        return self.track_class_map[track_id]
    
    def get_stale_track_ids(
        self,
        current_detections: list,
        tracked_objects: dict,
        frame_count: int
        ) -> list[int]:

        current_final_ids = {
            self.get_final_id(det.track_id) 
            for det in current_detections
        }
        
        absent_ids = set(tracked_objects.keys()) - current_final_ids

        stale_ids = []
        for final_id in absent_ids:
            tracked_obj = tracked_objects[final_id]
            last_seen = getattr(tracked_obj, 'last_seen_frame', tracked_obj.creation_frame)
            frames_since_seen = frame_count - last_seen
            
            if frames_since_seen > self.max_frames_lost:
                stale_ids.append(final_id)
        
        return stale_ids
        
    def resolve_detection_collisions(
        self,
        detections: list[Detection]
    ) -> dict[int, Detection]:
    
        final_id_map: dict[int, Detection] = {}
        
        for det in detections:
            final_id: int = self.get_final_id(det.track_id)
            
            # Se já existe detecção para este final_id, mantém a de maior confiança
            if final_id in final_id_map:
                if det.confidence > final_id_map[final_id].confidence:
                    final_id_map[final_id] = det
            else:
                final_id_map[final_id] = det
        
        return final_id_map