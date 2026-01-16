import json
from src.setup.config import (
    AppConfig, IOConfig, GeometryConfig, CalibrationMeshConfig, ModelConfig,
    ImageProcessingConfig, TrackingConfig, ColorConfig, AnnotationConfig,
    CalculationConfig, GeneralConfig
)

class ConfigLoader:

    def __init__(self, config_path='config.json'):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Arquivo de configuração não encontrado em: {config_path}")
        except json.JSONDecodeError:
            raise Exception(f"Erro ao decodificar o arquivo JSON: {config_path}")
        
        self.app_config = AppConfig(
            IO_CONFIG=IOConfig(**config_data['IO_CONFIG']),
            GEOMETRY_CONFIG=GeometryConfig(**config_data['GEOMETRY_CONFIG']),
            CALIBRATION_MESH_CONFIG=CalibrationMeshConfig(**config_data['CALIBRATION_MESH_CONFIG']),
            MODEL_CONFIG=ModelConfig(**config_data['MODEL_CONFIG']),
            IMAGE_PROCESSING_CONFIG=ImageProcessingConfig(**config_data['IMAGE_PROCESSING_CONFIG']),
            TRACKING_CONFIG=TrackingConfig(**config_data['TRACKING_CONFIG']),
            COLOR_CONFIG=ColorConfig(**config_data['COLOR_CONFIG']),
            ANNOTATION_CONFIG=AnnotationConfig(**config_data['ANNOTATION_CONFIG']),
            CALCULATION_CONFIG=CalculationConfig(**config_data['CALCULATION_CONFIG']),
            GENERAL_CONFIG=GeneralConfig(**config_data['GENERAL_CONFIG'])
        )

    def get_config(self) -> AppConfig:
        return self.app_config
