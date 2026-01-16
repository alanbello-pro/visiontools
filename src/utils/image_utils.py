import cv2
import numpy as np

from src.setup.config import ImageProcessingConfig

_gamma_cache = {}

def garantir_bgr(frame):

    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def melhorar_contraste(
    frame,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    gamma=1.6,
    bilateral_d=9,
    sigma_color=75,
    sigma_space=75
):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    if gamma not in _gamma_cache:
        inv_gamma = 1.0 / gamma
        _gamma_cache[gamma] = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    
    tabela = _gamma_cache[gamma]
    frame_gamma = cv2.LUT(frame_clahe, tabela)

    frame_suave = cv2.bilateralFilter(
        frame_gamma, d=bilateral_d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )

    return frame_suave

def preprocess_frame(frame, processing_config: ImageProcessingConfig):

    frame = garantir_bgr(frame)
    
    frame = melhorar_contraste(
        frame,
        clip_limit=processing_config.CLIP_LIMIT,
        tile_grid_size=tuple(processing_config.TILE_GRID_SIZE),
        gamma=processing_config.GAMMA,
        bilateral_d=processing_config.BILATERAL_D,
        sigma_color=processing_config.SIGMA_COLOR,
        sigma_space=processing_config.SIGMA_SPACE
    )
    return frame

def draw_timer_on_frame(
    frame: np.ndarray,
    frame_count: int,
    fps: float,
    font_scale: float,
    font_thickness: int,
    text_color_bgr: tuple,
    bg_color_bgr: tuple,
    margin_px: int,
    padding_px: int
) -> np.ndarray:
    frame_copy = frame.copy()
    
    if fps > 0:
        total_seconds = int(frame_count / fps)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        timer_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        timer_text = "00:00:00"

    (text_w, text_h), _ = cv2.getTextSize(
        timer_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    
    _, frame_w = frame_copy.shape[:2]
    x = frame_w - text_w - margin_px
    y = margin_px + text_h

    cv2.rectangle(
        frame_copy,
        (x - padding_px, y - text_h - padding_px),
        (x + text_w + padding_px, y + padding_px),
        bg_color_bgr,
        -1
    )

    cv2.putText(
        frame_copy, 
        timer_text, 
        (x, y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        text_color_bgr, 
        font_thickness, 
        cv2.LINE_AA
    )

    return frame_copy
