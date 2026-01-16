from src.models.tracked_object import TrackedObject
from src.models.data_models import Detection


class TrackDataCollector:

    def collect_data(
        self,
        tracked_obj: TrackedObject,
        det: Detection,
        frame_count: int,
    ) -> tuple[dict, dict]:
        registro_data = tracked_obj.to_csv_record(frame_count, det.confidence)
        annotation_data = tracked_obj.to_annotation_data()
        return registro_data, annotation_data
