from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class IOConfig:
    VIDEO_INPUT: str
    VIDEO_OUTPUT: str
    CSV_OUTPUT: str
    MALHA_EXPORT_FILENAME: str
    CRIAR_DIRETORIO_SAIDA: bool

@dataclass
class GeometryConfig:
    REGIAO_1: List[List[int]]
    REGIAO_CALIBRACAO_VELOCIDADE: List[List[int]]
    ZONA_DE_VALIDACAO: List[List[int]]
    MASCARAS_FAIXAS: Dict[str, List[List[int]]]

@dataclass
class CalibrationMeshConfig:
    NUM_LINHAS: int
    NUM_COLUNAS: int
    LINHAS_ANTES: int
    LINHAS_DEPOIS: int
    ESPACAMENTO_LINHAS_M: int
    ESPACAMENTO_COLUNAS_M: int

@dataclass
class ModelConfig:
    MODELO_YOLO: str
    TRACKZONE_CONF: float
    TRACKZONE_IOU: float
    TRACKZONE_CLASSES: List[int]
    TRACKZONE_DEVICE: str

@dataclass
class ImageProcessingConfig:
    CLIP_LIMIT: float
    TILE_GRID_SIZE: List[int]
    GAMMA: float
    BILATERAL_D: int
    SIGMA_COLOR: int
    SIGMA_SPACE: int
    ENABLE_PREPROCESSING: bool
    SHOW_FILTERS_IN_OUTPUT: bool


@dataclass
class TrackingConfig:
    MAX_FRAMES_LOST: int
    IOU_THRESHOLD: float
    KALMAN_PROCESS_NOISE: float
    KALMAN_MEASUREMENT_NOISE: float
    TRACKING_COLOR_WEIGHT: float

@dataclass
class ColorConfig:
    KMEANS_K_CLUSTERS: int
    KMEANS_MAX_ITER: int
    KMEANS_EPSILON: float
    KMEANS_ATTEMPTS: int
    RECALCULATION_INTERVAL: int
    BACKGROUND_IMAGE_PATH: Optional[str]
    COLOR_THRESHOLD: int

@dataclass
class AnnotationConfig:
    FONT_SCALE: float
    ID_FONT_SCALE: float
    FONT_THICKNESS: int
    SPEED_THRESH_LOW: int
    SPEED_THRESH_MED: int
    SPEED_COLOR_LOW: List[int]
    SPEED_COLOR_MED: List[int]
    SPEED_COLOR_HIGH: List[int]
    TRACK_COLOR_MULTIPLIER_B: int
    TRACK_COLOR_MULTIPLIER_G: int
    TRACK_COLOR_MULTIPLIER_R: int
    DETECTION_BOX_THICKNESS_PX: int
    LABEL_Y_OFFSET_PX: int
    SPEED_ANNOTATION_Y_OFFSET_PX: int
    SPEED_ANNOTATION_PADDING_PX: int
    SPEED_ANNOTATION_BG_COLOR_BGR: List[int]
    TIMER_FONT_SCALE: float
    TIMER_FONT_THICKNESS: int
    TIMER_TEXT_COLOR_BGR: List[int]
    TIMER_BG_COLOR_BGR: List[int]
    TIMER_MARGIN_PX: int
    TIMER_PADDING_PX: int
    CLASS_COLORS: Dict[str, List[int]]

@dataclass
class CalculationConfig:
    SPEED_MAX_KMH: int
    SPEED_MIN_KMH: int
    CSV_ROUND_PRECISION: int
    CSV_SEPARATOR: str
    MPS_TO_KMH_CONVERSION: float
    MMPS_TO_MPS_CONVERSION: float
    CSV_BATCH_SIZE: int

@dataclass
class GeneralConfig:
    PROGRESS_BAR_LENGTH: int
    VIDEO_WRITER_FOURCC: str
    SHOW_VIDEO_WINDOW: bool
    INFERENCE_RESIZE_WIDTH: int
    # Flags para funcionalidades opcionais
    ENABLE_SPEED_CALCULATION: bool = True
    ENABLE_VALIDATION_ZONE: bool = True
    ENABLE_CSV_OUTPUT: bool = True
    ENABLE_ANNOTATED_VIDEO: bool = True

@dataclass
class AppConfig:
    IO_CONFIG: IOConfig
    GEOMETRY_CONFIG: GeometryConfig
    CALIBRATION_MESH_CONFIG: CalibrationMeshConfig
    MODEL_CONFIG: ModelConfig
    IMAGE_PROCESSING_CONFIG: ImageProcessingConfig
    TRACKING_CONFIG: TrackingConfig
    COLOR_CONFIG: ColorConfig
    ANNOTATION_CONFIG: AnnotationConfig
    CALCULATION_CONFIG: CalculationConfig
    GENERAL_CONFIG: GeneralConfig