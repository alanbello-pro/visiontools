from typing import List
import numpy as np

from src.models.data_models import FrameTrackData


class OutputBuffer:
    
    def __init__(
        self, 
        csv_adapter=None, 
        video_adapter=None, 
        batch_size: int = 100
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size deve ser > 0, recebido: {batch_size}")
        
        self.csv_adapter = csv_adapter
        self.video_adapter = video_adapter
        self.batch_size = batch_size
        
        self.buffer: List[FrameTrackData] = []
        self.frame_count = 0
    
    def add(self, frame_data: FrameTrackData) -> None:

        self.buffer.append(frame_data)
        self.frame_count += 1
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self) -> None:
        if not self.buffer:
            return
        
        # Persiste CSV (se configurado)
        if self.csv_adapter:
            for frame_data in self.buffer:
                for registro in frame_data.registros_csv:
                    self.csv_adapter.save_record(registro)
        
        # Persiste vídeo (se configurado)
        if self.video_adapter:
            for frame_data in self.buffer:
                self.video_adapter.write_frame(frame_data.frame_anotado)
        
        # Limpa buffer
        self.buffer.clear()
    
    def close(self) -> None:
        self.flush()
        
        if self.csv_adapter:
            self.csv_adapter.close()
        
        if self.video_adapter:
            self.video_adapter.close()
    
    def get_stats(self) -> dict:
        return {
            'total_frames': self.frame_count,
            'buffer_size': len(self.buffer),
            'batch_size': self.batch_size
        }