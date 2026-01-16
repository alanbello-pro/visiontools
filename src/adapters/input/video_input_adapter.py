import cv2
import numpy as np
from typing import Iterator, Tuple

class VideoInputAdapter:
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise IOError(f"Erro ao abrir vídeo: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.fps <= 0:
            self.cap.release()
            raise ValueError(
                f"FPS inválido ({self.fps}) no vídeo '{video_path}'. "
                "O arquivo pode estar corrompido."
            )
        
        self._frame_count = 0
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        self._frame_count = 0
        
        while True:
            success, frame = self.cap.read()
            
            if not success:
                break
            
            yield self._frame_count, frame
            self._frame_count += 1
    
    def get_properties(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }
    
    def close(self) -> None:
        if self.cap:
            self.cap.release()
            print(f"📹 VideoCapture fechado: {self.video_path}")