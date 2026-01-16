import cv2
from typing import Tuple


class VideoSetup:

    @staticmethod
    def setup_capture(video_path: str) -> Tuple[cv2.VideoCapture, int, int, float]:

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Erro ao abrir o vídeo: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps is None or fps <= 0:
            cap.release()
            raise ValueError(
                f"FPS inválido ({fps}) no vídeo '{video_path}'. "
                "O arquivo pode estar corrompido."
            )
        
        return cap, width, height, fps
    
    @staticmethod
    def setup_writer(output_path: str, fourcc: str, fps: float, 
                     width: int, height: int) -> cv2.VideoWriter:
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height)
        )
        
        if not writer.isOpened():
            raise IOError(
                f"Erro ao criar vídeo de saída '{output_path}'. "
                "Verifique codec e permissões."
            )
        
        return writer