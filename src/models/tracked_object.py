class TrackedObject:
    def __init__(self, track_id: int, box: list, classe: str, frame_count: int):
        self.id = track_id
        self.box = box
        self.classe = classe
        self.creation_frame = frame_count
        self.last_seen_frame = frame_count
        self.dominant_color = None
        self.histogram = None
        self.last_feature_update = -1
        self.velocidade_kmh = None
        self.faixa = None

    def to_csv_record(self, frame_id: int, confidence: float) -> dict:
        return {
            'frame_id': frame_id,
            'track_id': self.id,
            'classe': self.classe,
            'confianca': confidence,
            'box_x1': self.box[0],
            'box_y1': self.box[1],
            'box_x2': self.box[2],
            'box_y2': self.box[3],
            'velocidade_kmh': self.velocidade_kmh if self.velocidade_kmh is not None else 0.0,
            'faixa': self.faixa if self.faixa else '',
            'cor_dominante_bgr': self.dominant_color
        }

    def to_annotation_data(self) -> dict:
        return {
            'track_id': self.id,
            'classe': self.classe,
            'box': self.box,
            'velocidade': self.velocidade_kmh
        }