from typing import List
from src.models.data_models import Detection

class TrackZoneAdapter:
    def __init__(self, trackzone_instance, model):
        self.trackzone = trackzone_instance
        self.model = model
        self.model_names = getattr(self.model, 'names', {})
        if not self.model_names:
            raise ValueError("O modelo YOLO carregado não contém o atributo 'names' com os nomes das classes. Verifique se o arquivo do modelo está correto e completo.")

    def track(self, frame) -> List[Detection]:
        self.trackzone(frame)
        if not (hasattr(self.trackzone, 'track_ids') and self.trackzone.track_ids is not None and
                hasattr(self.trackzone, 'boxes') and self.trackzone.boxes is not None and
                hasattr(self.trackzone, 'clss') and self.trackzone.clss is not None and
                hasattr(self.trackzone, 'confs') and self.trackzone.confs is not None):
            return []

        num_tracks = len(self.trackzone.track_ids)
        if not (len(self.trackzone.boxes) == num_tracks and len(self.trackzone.clss) == num_tracks and len(self.trackzone.confs) == num_tracks):
            print(f"⚠️ Aviso: Dados do tracker inconsistentes - IDs:{len(self.trackzone.track_ids)}, "
                  f"Boxes:{len(self.trackzone.boxes)}, Classes:{len(self.trackzone.clss)}, "
                  f"Confs:{len(self.trackzone.confs)}")
            return []
        detections = []

        for i in range(num_tracks):
            track_id = int(self.trackzone.track_ids[i])
            class_id = int(self.trackzone.clss[i])
            class_name = self.model_names.get(class_id, f"class_{class_id}")

            box_array = self.trackzone.boxes[i].cpu().numpy().copy()
            if box_array.shape != (4,):
                print(f"⚠️ Aviso [track_id={track_id}]: Box com formato inválido ignorada: {box_array.shape}")
                continue

            conf_value = self.trackzone.confs[i]
            if hasattr(conf_value, 'item'):
                if conf_value.numel() != 1:
                    print(f"⚠️ Aviso [track_id={track_id}]: Tensor de confiança com formato inválido ignorado: {conf_value.shape}")
                    continue
                confidence = conf_value.item()
            else:
                try:
                    confidence = float(conf_value)
                except (ValueError, TypeError):
                    print(f"⚠️ Aviso [track_id={track_id}]: Valor de confiança não numérico ignorado: {conf_value}")
                    continue

            detections.append(Detection(
                track_id=track_id,
                box=box_array,
                class_name=class_name,
                confidence=confidence
            ))
        return detections