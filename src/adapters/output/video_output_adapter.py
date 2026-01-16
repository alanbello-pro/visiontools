import cv2
import numpy as np
from typing import Tuple

class VideoOutputAdapter:
    
    def __init__(
        self,
        filepath: str,
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: str = "mp4v"
    ):
        
        self.filepath = filepath
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = fourcc
        
        # Validações
        if fps <= 0:
            raise ValueError(f"FPS deve ser > 0, recebido: {fps}")
        
        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError(f"frame_size inválido: {frame_size}")
        
        # Cria VideoWriter
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(
            filepath,
            fourcc_code,
            fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise IOError(
                f"Erro ao criar vídeo '{filepath}'. "
                f"Verifique codec '{fourcc}' e permissões."
            )
        
        self.frame_count = 0
    
    def write_frame(self, frame: np.ndarray) -> None:
        
        if not self.writer or not self.writer.isOpened():
            raise RuntimeError("VideoWriter não está aberto")
        
        # Valida dimensões
        if frame.shape[:2][::-1] != self.frame_size:
            raise ValueError(
                f"Frame com dimensões incorretas. "
                f"Esperado: {self.frame_size}, "
                f"Recebido: {frame.shape[:2][::-1]}"
            )
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def close(self) -> None:
        """Fecha writer de vídeo."""
        if self.writer:
            self.writer.release()
            print(f"🎥 Vídeo salvo: {self.filepath} ({self.frame_count} frames)")