import cv2
import numpy as np
import math
from typing import Optional

from src.setup.config import CalculationConfig, TrackingConfig

class SpeedCalculator:
    def __init__(self, fps: float, matriz_perspectiva: np.ndarray, calculation_config: CalculationConfig, kalman_config: TrackingConfig):
        self.fps = fps
        self.filtros_kalman = {}
        self.update_counts = {} 
        self.calculation_config = calculation_config
        self.kalman_config = kalman_config
        self.matriz_perspectiva = matriz_perspectiva

    def _criar_filtro_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        dt = 1.0 / self.fps
        kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.kalman_config.KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.kalman_config.KALMAN_MEASUREMENT_NOISE
        return kf

    def update_position(self, track_id: int, ponto_central: tuple):

        if ponto_central is None:
            return
        
        px, py = ponto_central
        if not (math.isfinite(px) and math.isfinite(py)):
            return

        ponto_np = np.array([[[ponto_central[0], ponto_central[1]]]], dtype=np.float32)
        ponto_transformado = cv2.perspectiveTransform(ponto_np, self.matriz_perspectiva)[0][0]

        if not np.all(np.isfinite(ponto_transformado)):
            return

        medicao = np.array([ponto_transformado[0], ponto_transformado[1]], dtype=np.float32)

        if track_id not in self.filtros_kalman:
            kf = self._criar_filtro_kalman()
            kf.statePost = np.array([medicao[0], medicao[1], 0, 0], dtype=np.float32)
            self.filtros_kalman[track_id] = kf
            self.update_counts[track_id] = 1
        
        kf = self.filtros_kalman[track_id]

        self.update_counts[track_id] = self.update_counts.get(track_id, 0) + 1

        kf.predict()
        kf.correct(medicao)

    def get_speed(self, track_id: int) -> Optional[float]:
        min_updates_for_stable_speed = 3
        if track_id not in self.filtros_kalman or self.update_counts.get(track_id, 0) < min_updates_for_stable_speed:
            return None

        kf = self.filtros_kalman[track_id]
        _, _, vx, vy = kf.statePost.flatten()

        velocidade_mmps = math.hypot(vx, vy)
        velocidade_mps = velocidade_mmps / self.calculation_config.MMPS_TO_MPS_CONVERSION
        velocidade_kmh = velocidade_mps * self.calculation_config.MPS_TO_KMH_CONVERSION
        
        if self.calculation_config.SPEED_MIN_KMH < velocidade_kmh < self.calculation_config.SPEED_MAX_KMH:
            return velocidade_kmh
        
        return None

    def remove_filter(self, track_id: int):
        self.filtros_kalman.pop(track_id, None)
        self.update_counts.pop(track_id, None)
