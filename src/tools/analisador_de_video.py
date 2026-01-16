from ultralytics import YOLO
import traceback
import cv2
import argparse
import logging
import os

from src.ui.frame_annotator import FrameAnnotator
from src.ui.ui import print_progress
from src.adapters.trackzone_adapter import TrackZoneAdapter

# Imports de adapters de I/O
from src.adapters.input.video_input_adapter import VideoInputAdapter
from src.adapters.output.output_buffer import OutputBuffer
from src.adapters.output.csv_output_adapter import CSVOutputAdapter
from src.adapters.output.video_output_adapter import VideoOutputAdapter
from src.adapters.csv_saver import exportar_malha_para_csv

# Imports de pipelines e models
from src.pipelines.tracking_manager import TrackingManager
from src.pipelines.speed_calculator import SpeedCalculator
from src.pipelines.track_processor import TrackProcessor
from src.pipelines.track_data_collector import TrackDataCollector
from src.pipelines.track_lifecycle_manager import TrackLifecycleManager
from src.pipelines.frame_processor import FramePreprocessor
from src.pipelines import tracking_helpers
from src.models.data_models import FrameTrackData

# Imports de setup e utils
from src.setup.components import initialize_trackzone, determine_yolo_device
from src.setup.config_loader import ConfigLoader
from src.setup.background import BackgroundSetup
from src.setup.calibration import CalibrationSetup
from src.setup.paths import PathManager
from src.setup.resources import ResourceManager
from src.utils.filesystem import verificar_e_criar_diretorios_input

logging.getLogger("ultralytics").setLevel(logging.WARNING)


def parse_cli_arguments():
    parser = argparse.ArgumentParser(
        description="Analisador de Vídeo com YOLO - Tracking e Análise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Modo padrão (usa todas as configurações de config.json)
  python -m src.tools.analisador_de_video
  
  # Gerar apenas CSV (sem vídeo anotado)
  python -m src.tools.analisador_de_video --only-csv
  
  # Gerar apenas vídeo anotado (sem CSV)
  python -m src.tools.analisador_de_video --only-video
  
  # Exibir janela durante processamento
  python -m src.tools.analisador_de_video --show
  
  # Não exibir janela (útil para processamento em background)
  python -m src.tools.analisador_de_video --no-show
  
  # Combinações
  python -m src.tools.analisador_de_video --only-video --show
  python -m src.tools.analisador_de_video --no-video --no-csv  # Apenas tracking
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Caminho para o arquivo de configuração (padrão: config.json)'
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        '--only-csv',
        action='store_true',
        help='Gerar apenas arquivo CSV (desabilita vídeo anotado)'
    )
    output_group.add_argument(
        '--only-video',
        action='store_true',
        help='Gerar apenas vídeo anotado (desabilita CSV)'
    )

    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Desabilitar geração de CSV'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Desabilitar geração de vídeo anotado'
    )

    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        '--show',
        action='store_true',
        help='Exibir janela de vídeo durante processamento'
    )
    display_group.add_argument(
        '--no-show',
        action='store_true',
        help='Não exibir janela de vídeo'
    )
    
    return parser.parse_args()


def apply_cli_overrides(app_config, args):
    if args.only_csv:
        app_config.GENERAL_CONFIG.ENABLE_CSV_OUTPUT = True
        app_config.GENERAL_CONFIG.ENABLE_ANNOTATED_VIDEO = False
        print("Modo CLI: Apenas CSV habilitado")
        
    elif args.only_video:
        app_config.GENERAL_CONFIG.ENABLE_CSV_OUTPUT = False
        app_config.GENERAL_CONFIG.ENABLE_ANNOTATED_VIDEO = True
        print("Modo CLI: Apenas vídeo anotado habilitado")
    
    # Handle individual toggles
    if args.no_csv:
        app_config.GENERAL_CONFIG.ENABLE_CSV_OUTPUT = False
        print("Modo CLI: CSV desabilitado")
        
    if args.no_video:
        app_config.GENERAL_CONFIG.ENABLE_ANNOTATED_VIDEO = False
        print("Modo CLI: Vídeo anotado desabilitado")
    
    # Handle display window
    if args.show:
        app_config.GENERAL_CONFIG.SHOW_VIDEO_WINDOW = True
        print("Modo CLI: Exibição de janela habilitada")
        
    elif args.no_show:
        app_config.GENERAL_CONFIG.SHOW_VIDEO_WINDOW = False
        print("Modo CLI: Exibição de janela desabilitada")
    
    return app_config


def initialize_components(project_root: str, config_path: str, cli_args=None) -> dict:
    print("\n" + "="*60)
    print("INICIALIZANDO COMPONENTES DO PIPELINE")
    print("="*60)

    verificar_e_criar_diretorios_input(project_root)
    app_config = ConfigLoader(config_path=config_path).get_config()
    
    if cli_args:
        app_config = apply_cli_overrides(app_config, cli_args)

    # Torna os caminhos de I/O absolutos
    app_config.IO_CONFIG.VIDEO_INPUT = os.path.join(project_root, app_config.IO_CONFIG.VIDEO_INPUT)
    app_config.MODEL_CONFIG.MODELO_YOLO = os.path.join(project_root, app_config.MODEL_CONFIG.MODELO_YOLO)
    if app_config.COLOR_CONFIG.BACKGROUND_IMAGE_PATH:
        app_config.COLOR_CONFIG.BACKGROUND_IMAGE_PATH = os.path.join(project_root, app_config.COLOR_CONFIG.BACKGROUND_IMAGE_PATH)

    resources = ResourceManager()
    if app_config.MODEL_CONFIG.TRACKZONE_DEVICE is None:
        app_config.MODEL_CONFIG.TRACKZONE_DEVICE = determine_yolo_device()
    print(f"Dispositivo YOLO: {app_config.MODEL_CONFIG.TRACKZONE_DEVICE}")

    path_manager = PathManager(project_root, app_config.IO_CONFIG)
    video_out, csv_out, malha_out = path_manager.setup_paths()

    print("\nConfigurando input de vídeo...")
    video_input = VideoInputAdapter(app_config.IO_CONFIG.VIDEO_INPUT)
    resources.video_input = video_input
    
    props = video_input.get_properties()
    width, height, fps, total_frames = props['width'], props['height'], props['fps'], props['total_frames']
    print(f"Vídeo carregado: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

    print("\nCarregando modelo YOLO...")
    model = YOLO(app_config.MODEL_CONFIG.MODELO_YOLO)
    trackzone = initialize_trackzone(model, app_config.GEOMETRY_CONFIG, app_config.MODEL_CONFIG)
    trackzone_adapter = TrackZoneAdapter(trackzone, model)
    print("Modelo YOLO carregado")

    print("\n Configurando calibração de perspectiva...")
    matriz_h, malha_pixels = CalibrationSetup.setup_calibration(
        app_config.GEOMETRY_CONFIG, app_config.CALIBRATION_MESH_CONFIG
    )
    exportar_malha_para_csv(
        malha_pixels, app_config.CALIBRATION_MESH_CONFIG.ESPACAMENTO_LINHAS_M,
        app_config.CALIBRATION_MESH_CONFIG.ESPACAMENTO_COLUNAS_M, malha_out,
        app_config.CALCULATION_CONFIG.CSV_ROUND_PRECISION, app_config.CALCULATION_CONFIG.CSV_SEPARATOR
    )
    print(f"Malha de calibração exportada: {malha_out}")

    print("\n Inicializando tracking manager...")
    tracking_manager = TrackingManager(
        max_frames_lost=app_config.TRACKING_CONFIG.MAX_FRAMES_LOST,
        iou_threshold=app_config.TRACKING_CONFIG.IOU_THRESHOLD,
        color_weight=app_config.TRACKING_CONFIG.TRACKING_COLOR_WEIGHT
    )
    
    bg_color = BackgroundSetup.setup_background(app_config.COLOR_CONFIG)

    speed_calc = None
    if app_config.GENERAL_CONFIG.ENABLE_SPEED_CALCULATION:
        print("Habilitando cálculo de velocidade...")
        speed_calc = SpeedCalculator(
            fps=fps, matriz_perspectiva=matriz_h,
            calculation_config=app_config.CALCULATION_CONFIG,
            kalman_config=app_config.TRACKING_CONFIG
        )
    
    track_processor = None
    if app_config.GENERAL_CONFIG.ENABLE_VALIDATION_ZONE:
        print("Habilitando zona de validação...")
        track_processor = TrackProcessor(
            geometry_config=app_config.GEOMETRY_CONFIG,
            color_config=app_config.COLOR_CONFIG,
            background_color_to_ignore=bg_color,
            speed_calc=speed_calc
        )
    
    track_data_collector = None
    if app_config.GENERAL_CONFIG.ENABLE_CSV_OUTPUT or app_config.GENERAL_CONFIG.ENABLE_ANNOTATED_VIDEO:
        track_data_collector = TrackDataCollector()

    track_lifecycle_manager = TrackLifecycleManager(
        tracking_manager=tracking_manager,
        geometry_config=app_config.GEOMETRY_CONFIG,
        speed_calc=speed_calc,
        track_processor=track_processor,
        track_data_collector=track_data_collector
    )
    
    frame_annotator = None
    if app_config.GENERAL_CONFIG.ENABLE_ANNOTATED_VIDEO:
        print("Habilitando anotação de frames...")
        frame_annotator = FrameAnnotator(annotator_config=app_config.ANNOTATION_CONFIG)

    csv_adapter, video_adapter = None, None
    
    if app_config.GENERAL_CONFIG.ENABLE_CSV_OUTPUT:
        print(f"Configurando saída CSV: {csv_out}")
        csv_adapter = CSVOutputAdapter(
            filepath=csv_out,
            batch_size=app_config.CALCULATION_CONFIG.CSV_BATCH_SIZE
        )
        resources.csv_adapter = csv_adapter
    
    if app_config.GENERAL_CONFIG.ENABLE_ANNOTATED_VIDEO:
        print(f"Configurando saída de vídeo: {video_out}")
        video_adapter = VideoOutputAdapter(
            filepath=video_out,
            fps=fps,
            frame_size=(width, height),
            fourcc=app_config.GENERAL_CONFIG.VIDEO_WRITER_FOURCC
        )
        resources.video_adapter = video_adapter
    
    output_buffer = OutputBuffer(
        csv_adapter=csv_adapter,
        video_adapter=video_adapter,
        batch_size=100
    )
    resources.output_buffer = output_buffer
    resize_width = app_config.GENERAL_CONFIG.INFERENCE_RESIZE_WIDTH
    original_size = (width, height)
    inference_size = original_size
    
    if resize_width > 0 and width > 0:
        scale = resize_width / width
        inference_size = (resize_width, int(height * scale))
        print(f"Redimensionamento para inferência: {original_size} → {inference_size}")
    
    frame_preprocessor = FramePreprocessor(
        original_size=original_size,
        inference_size=inference_size,
        image_processing_config=app_config.IMAGE_PROCESSING_CONFIG
    )
    
    print("\n" + "="*60)
    print("TODOS OS COMPONENTES INICIALIZADOS COM SUCESSO")
    print("="*60 + "\n")
    
    return {
        'app_config': app_config,
        'resources': resources,
        'video_input': video_input,
        'video_props': props,
        'model': model,
        'trackzone_adapter': trackzone_adapter,
        'matriz_h': matriz_h,
        'malha_pixels': malha_pixels,
        'tracking_manager': tracking_manager,
        'track_lifecycle_manager': track_lifecycle_manager,
        'frame_preprocessor': frame_preprocessor,
        'speed_calc': speed_calc,
        'track_processor': track_processor,
        'track_data_collector': track_data_collector,
        'frame_annotator': frame_annotator,
        'output_buffer': output_buffer,
        'csv_adapter': csv_adapter,
        'video_adapter': video_adapter,
        'video_out': video_out,
        'csv_out': csv_out,
        'malha_out': malha_out
    }


def process_frame(frame_id: int, frame, components: dict) -> FrameTrackData:
    config = components['app_config']
    
    inference_frame = components['frame_preprocessor'].prepare_for_inference(frame)
    frame_anotado = components['frame_preprocessor'].prepare_for_annotation(
        frame, show_filters=config.IMAGE_PROCESSING_CONFIG.SHOW_FILTERS_IN_OUTPUT
    )
    
    detections = components['trackzone_adapter'].track(inference_frame.copy())
    scaled_detections = tracking_helpers.scale_detections(
        detections, components['frame_preprocessor'].scale_factor
    )
    
    current_tracks, histogram_map = tracking_helpers.prepare_tracking_data(
        scaled_detections, frame
    )
    
    components['tracking_manager'].update_tracks(current_tracks, frame_id)
    
    csv_records, annotation_data = components['track_lifecycle_manager'].process_all_tracks(
        scaled_detections=scaled_detections,
        frame=frame,
        frame_count=frame_id,
        histogram_map=histogram_map
    )
    
    if components['frame_annotator']:
        if annotation_data:
            for data in annotation_data:
                components['frame_annotator'].desenhar_track(frame_anotado, **data)
        frame_anotado = components['frame_annotator'].draw_video_timer(
            frame_anotado, frame_id, components['video_props']['fps']
        )
    components['track_lifecycle_manager'].clear_stale_objects(frame_id, scaled_detections)
    
    return FrameTrackData(
        frame_id=frame_id,
        frame_anotado=frame_anotado,
        registros_csv=csv_records,
        fps=components['video_props']['fps']
    )


def run_processing_loop(components: dict) -> None:
    config = components['app_config']
    total_frames = components['video_props']['total_frames']
    fps = components['video_props']['fps']
    
    print("INICIANDO PROCESSAMENTO DE VÍDEO")
    print("=" * 60)
    print(f"Total de frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print("Pressione 'q' para interromper\n")
    
    try:
        for frame_id, frame in components['video_input']:
            frame_data = process_frame(frame_id, frame, components)
            components['output_buffer'].add(frame_data)
            num_unique = components['csv_adapter'].get_unique_track_count() if components['csv_adapter'] else 0
            print_progress(
                frame_id, total_frames, fps, num_unique,
                config.GENERAL_CONFIG.PROGRESS_BAR_LENGTH
            )
            if config.GENERAL_CONFIG.SHOW_VIDEO_WINDOW:
                cv2.imshow("Frame Processado", frame_data.frame_anotado)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\n Processamento interrompido pelo usuário (tecla 'q').")
                    break
        
        print("\n Processamento concluído com sucesso!")
        
    except KeyboardInterrupt:
        print("\n\n Processamento interrompido pelo usuário (Ctrl+C).")
        raise


def main():
    components = None
    
    try:
        # Define o caminho raiz do projeto (3 níveis acima de src/tools/analisador_de_video.py)
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        args = parse_cli_arguments()

        # Garante que o caminho do config seja absoluto
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(PROJECT_ROOT, config_path)

        components = initialize_components(
            project_root=PROJECT_ROOT,
            config_path=config_path, 
            cli_args=args
        )

        run_processing_loop(components)
        
    except KeyboardInterrupt:
        print("\n\n Processamento interrompido pelo usuário.")
        
    except Exception as e:
        print(f"\n Erro durante o processamento:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\nTraceback completo:")
        traceback.print_exc()
        raise
        
    finally:
        if components:
            print("\n🧹 Liberando recursos...")

            if components['track_lifecycle_manager']:
                components['track_lifecycle_manager'].cleanup_all_tracking()

            components['resources'].cleanup(components.get('app_config'))
            
            print("Recursos liberados com sucesso")


if __name__ == "__main__":
    main()