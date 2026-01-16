import csv

class CSVOutputAdapter:
    
    FIELDNAMES = [
        'frame_id', 'track_id', 'classe', 'confianca',
        'box_x1', 'box_y1', 'box_x2', 'box_y2', 
        'velocidade_kmh', 'faixa', 'cor_dominante_bgr'
    ]
    
    def __init__(self, filepath: str, batch_size: int = 50):
        self.filepath = filepath
        self.batch_size = batch_size
        self.buffer = []
        self.saved_track_ids = set()  # Para estatísticas
        
        try:
            self.csvfile = open(filepath, 'w', newline='', encoding='utf-8')
            self.writer = csv.DictWriter(self.csvfile, fieldnames=self.FIELDNAMES)
            self.writer.writeheader()
        except IOError as e:
            raise IOError(f"Erro ao criar arquivo CSV '{filepath}': {e}")
    
    def save_record(self, record: dict) -> None:
        if not self.writer:
            raise RuntimeError("CSV adapter não está inicializado")
        
        # Processa e valida registro
        processed = self._process_record(record)
        
        # Adiciona ao buffer interno
        self.buffer.append(processed)
        
        # Atualiza estatísticas
        track_id = int(processed.get('track_id', 0))
        if track_id:
            self.saved_track_ids.add(track_id)
        
        # Flush automático se buffer cheio
        if len(self.buffer) >= self.batch_size:
            self._flush_internal()
    
    def _process_record(self, record: dict) -> dict:

        processed = {
            'frame_id': int(self._convert_value(record.get('frame_id', 0))),
            'track_id': int(self._convert_value(record.get('track_id', 0))),
            'classe': str(record.get('classe', '')),
            'confianca': round(self._convert_value(record.get('confianca', 0.0)), 4),
            'box_x1': round(self._convert_value(record.get('box_x1', 0.0)), 2),
            'box_y1': round(self._convert_value(record.get('box_y1', 0.0)), 2),
            'box_x2': round(self._convert_value(record.get('box_x2', 0.0)), 2),
            'box_y2': round(self._convert_value(record.get('box_y2', 0.0)), 2),
            'velocidade_kmh': self._format_velocity(record.get('velocidade_kmh')),
            'faixa': str(record.get('faixa', '')),
            'cor_dominante_bgr': self._format_color(record.get('cor_dominante_bgr'))
        }
        
        return processed
    
    def _convert_value(self, value):
        if hasattr(value, 'item'):  # Tensor PyTorch
            return float(value.item())
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    
    def _format_velocity(self, velocity) -> str:
        if velocity is None:
            return ''
        return str(round(self._convert_value(velocity), 2))
    
    def _format_color(self, color) -> str:
        if color is None or not isinstance(color, (list, tuple)):
            return ''
        
        if len(color) != 3:
            return ''
        
        return f"{color[0]},{color[1]},{color[2]}"
    
    def _flush_internal(self) -> None:
        if not self.buffer:
            return
        
        try:
            self.writer.writerows(self.buffer)
            self.buffer.clear()
        except IOError as e:
            print(f"⚠️ Erro ao escrever batch no CSV: {e}")
    
    def get_unique_track_count(self) -> int:
        return len(self.saved_track_ids)
    
    def close(self) -> None:
        self._flush_internal()
        
        if self.csvfile:
            self.csvfile.close()
            print(f"📄 CSV salvo: {self.filepath}")