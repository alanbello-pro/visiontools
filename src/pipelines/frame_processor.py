import cv2
import numpy as np
from typing import Tuple
from src.utils.image_utils import preprocess_frame
from src.setup.config import ImageProcessingConfig


class FramePreprocessor:
  
    def __init__(
        self,
        original_size: Tuple[int, int],
        inference_size: Tuple[int, int],
        image_processing_config: ImageProcessingConfig
    ):
        self.original_size = original_size
        self.inference_size = inference_size
        self.config = image_processing_config
        
        # Calcula escala uma vez
        orig_w = original_size[0]
        inf_w = inference_size[0]
        self.scale_factor = inf_w / orig_w if orig_w > 0 else 1.0
        self.needs_resize = (self.scale_factor != 1.0)
    
    def prepare_for_inference(self, frame: np.ndarray) -> np.ndarray:
        if self.needs_resize:
            frame = cv2.resize(frame, self.inference_size, interpolation=cv2.INTER_AREA)

        if self.config.ENABLE_PREPROCESSING:
            frame = preprocess_frame(frame, self.config)
        
        return frame
    
    def prepare_for_annotation(self, frame: np.ndarray, show_filters: bool = False) -> np.ndarray:

        frame_copy = frame.copy()
        
        if show_filters and self.config.ENABLE_PREPROCESSING:
            frame_copy = preprocess_frame(frame_copy, self.config)
        
        return frame_copy
