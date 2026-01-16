import numpy as np
from typing import Dict, Optional, Tuple, List

from src.models.tracked_object import TrackedObject
from src.models.data_models import Detection
from src.utils.geometry import check_bbox_in_masks
from src.pipelines.tracking_manager import TrackingManager
from src.pipelines.speed_calculator import SpeedCalculator
from src.pipelines.track_processor import TrackProcessor
from src.pipelines.track_data_collector import TrackDataCollector
from src.setup.config import GeometryConfig


class TrackLifecycleManager:

    def __init__(
        self,
        tracking_manager: TrackingManager,
        geometry_config: GeometryConfig,
        speed_calc: Optional[SpeedCalculator] = None,
        track_processor: Optional[TrackProcessor] = None,
        track_data_collector: Optional[TrackDataCollector] = None,
    ):
        self.tracking_manager = tracking_manager
        self.geometry_config = geometry_config
        self.speed_calc = speed_calc
        self.track_processor = track_processor
        self.track_data_collector = track_data_collector

        self.tracked_objects: Dict[int, TrackedObject] = {}

    def process_all_tracks(
        self,
        scaled_detections: list[Detection],
        frame: np.ndarray,
        frame_count: int,
        histogram_map: dict[int, np.ndarray],
    ) -> Tuple[List[dict], List[dict]]:

        final_detections = self.tracking_manager.resolve_detection_collisions(
            scaled_detections
        )

        histogram_by_final_id = {
            final_id: histogram_map[det.track_id]
            for final_id, det in final_detections.items()
            if det.track_id in histogram_map
        }

        all_csv_records = []
        all_annotation_data = []

        for final_id, det in final_detections.items():
            csv_record, annotation_data = self._process_single_track(
                final_id, det, frame, frame_count,
                histogram_by_final_id
            )
            if csv_record:
                all_csv_records.append(csv_record)
            if annotation_data:
                all_annotation_data.append(annotation_data)
        
        return all_csv_records, all_annotation_data

    def _process_single_track(
        self,
        final_id: int,
        det: Detection,
        frame: np.ndarray,
        frame_count: int,
        histogram_by_final_id: dict[int, np.ndarray],
    ) -> Tuple[Optional[dict], Optional[dict]]:
        
        tracked_obj = self._get_or_create_tracked_object(
            final_id, det, det.box, frame_count
        )

        self._update_continuous_tracking(tracked_obj, det.box, final_id, frame_count)

        if self.track_processor and self._is_in_validation_zone(tracked_obj.box):

            self.track_processor.process_track(
                tracked_obj, frame, frame_count, histogram_by_final_id
            )

            if self.track_data_collector:
                return self.track_data_collector.collect_data(
                    tracked_obj, det, frame_count
                )
        else:

            tracked_obj.velocidade_kmh = None

        return None, None

    def _get_or_create_tracked_object(self, final_id: int, det: Detection, box: np.ndarray, frame_count: int) -> TrackedObject:
        if final_id not in self.tracked_objects:
            classe = self.tracking_manager.get_or_set_class(final_id, det.class_name)
            self.tracked_objects[final_id] = TrackedObject(
                final_id, box, classe, frame_count
            )
        return self.tracked_objects[final_id]

    def _update_continuous_tracking(
        self, tracked_obj: TrackedObject, box: np.ndarray, final_id: int, frame_count: int
    ) -> None:
        
        tracked_obj.box = box.copy()
        tracked_obj.last_seen_frame = frame_count

        if self.speed_calc:
            x_centro = (box[0] + box[2]) / 2
            y_base = box[3]
            ponto_central = (x_centro, y_base)
            self.speed_calc.update_position(final_id, ponto_central)

    def _is_in_validation_zone(self, box: np.ndarray) -> bool:

        if not self.geometry_config.ZONA_DE_VALIDACAO:
            return False
        validation_mask = {'validation': self.geometry_config.ZONA_DE_VALIDACAO}
        return check_bbox_in_masks(box, validation_mask)

    def clear_stale_objects(self, frame_count: int, current_detections: list[Detection]) -> None:

        stale_ids = self.tracking_manager.get_stale_track_ids(
            current_detections=current_detections,
            tracked_objects=self.tracked_objects,
            frame_count=frame_count
        )

        for stale_id in stale_ids:
            if stale_id in self.tracked_objects:
                del self.tracked_objects[stale_id]
            if self.speed_calc:
                self.speed_calc.remove_filter(stale_id)

    def cleanup_all_tracking(self) -> None:

        if self.speed_calc:
            for track_id in list(self.tracked_objects.keys()):
                self.speed_calc.remove_filter(track_id)
        self.tracked_objects.clear()