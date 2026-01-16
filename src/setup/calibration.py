import cv2
import numpy as np

from src.pipelines.perspective_calibrator import PerspectiveCalibrator


class CalibrationSetup:
    
    @staticmethod
    def setup_calibration(geometry_config, mesh_config):
        print("Iniciando calibração de perspectiva...")
        
        try:
            calibrator = PerspectiveCalibrator(
                trapezio=geometry_config.REGIAO_CALIBRACAO_VELOCIDADE
            )
            
            malha_pixels = calibrator.calcular_malha(
                num_linhas=mesh_config.NUM_LINHAS,
                num_colunas=mesh_config.NUM_COLUNAS,
                linhas_antes=mesh_config.LINHAS_ANTES,
                linhas_depois=mesh_config.LINHAS_DEPOIS
            )
        except ValueError as e:
            raise RuntimeError(f"Falha ao gerar malha de calibração: {e}")

        if not malha_pixels or not malha_pixels[0]:
            raise RuntimeError("Malha de calibração vazia")

        total_linhas = len(malha_pixels)
        total_colunas = len(malha_pixels[0])
        
        pontos_reais = [
            [j * mesh_config.ESPACAMENTO_COLUNAS_M, 
             i * mesh_config.ESPACAMENTO_LINHAS_M]
            for i in range(total_linhas)
            for j in range(total_colunas)
        ]
        
        pontos_pixels = [p for linha in malha_pixels for p in linha]

        matriz_h, _ = cv2.findHomography(
            np.float32(pontos_pixels), 
            np.float32(pontos_reais)
        )
        
        if matriz_h is None:
            raise RuntimeError(
                "Falha ao calcular homografia. "
                "Verifique se os pontos não são colineares."
            )
        
        print("Matriz de homografia calculada com sucesso.")
        return matriz_h, malha_pixels
