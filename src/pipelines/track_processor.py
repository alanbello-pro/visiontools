import numpy as np
from typing import Optional

from src.models.tracked_object import TrackedObject
from src.utils.geometry import check_bbox_in_masks
from src.pipelines.feature_extractor import FeatureExtractor
from src.pipelines.speed_calculator import SpeedCalculator
from src.setup.config import GeometryConfig, ColorConfig


class TrackProcessor:
    def __init__(
        self,
        geometry_config: GeometryConfig,
        color_config: ColorConfig,
        background_color_to_ignore: tuple,
        speed_calc: Optional[SpeedCalculator] = None,
    ):
        self.geometry_config = geometry_config
        self.color_config = color_config
        self.background_color_to_ignore = background_color_to_ignore
        self.speed_calc = speed_calc

    def process_track(
        self,
        tracked_obj: TrackedObject,
        frame: np.ndarray,
        frame_count: int,
        histogram_by_id: dict[int, np.ndarray],
    ) -> None:
        self._update_lane(tracked_obj)
        self._update_histogram(tracked_obj, histogram_by_id)
        self._update_features(tracked_obj, frame, frame_count)
        self._update_speed(tracked_obj)

    def _update_lane(self, tracked_obj: TrackedObject) -> None:
        tracked_obj.faixa = check_bbox_in_masks(
            tracked_obj.box,
            self.geometry_config.MASCARAS_FAIXAS
        )

    def _update_histogram(
        self,
        tracked_obj: TrackedObject,
        histogram_by_id: dict[int, np.ndarray]
    ) -> None:
        
        if tracked_obj.id in histogram_by_id:
            tracked_obj.histogram = histogram_by_id[tracked_obj.id]

    def _update_features(
        self, tracked_obj: TrackedObject, frame: np.ndarray, frame_count: int
    ) -> None:
        is_first_update = tracked_obj.last_feature_update == -1
        interval_passed = (is_first_update is False) and \
                          ((frame_count - tracked_obj.last_feature_update) >= self.color_config.RECALCULATION_INTERVAL)

        if is_first_update or interval_passed:
            new_color = FeatureExtractor.get_dominant_color(
                frame, tracked_obj.box,
                k=self.color_config.KMEANS_K_CLUSTERS,
                max_iter=self.color_config.KMEANS_MAX_ITER,
                epsilon=self.color_config.KMEANS_EPSILON,
                attempts=self.color_config.KMEANS_ATTEMPTS,
                ignore_colors=self.background_color_to_ignore,
                color_threshold=self.color_config.COLOR_THRESHOLD
            )

            if new_color is not None:
                tracked_obj.dominant_color = new_color

            tracked_obj.last_feature_update = frame_count

    def _update_speed(self, tracked_obj: TrackedObject) -> None:

        if self.speed_calc:
            tracked_obj.velocidade_kmh = self.speed_calc.get_speed(tracked_obj.id)
