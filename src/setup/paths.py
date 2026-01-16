import os
from typing import Tuple
from src.utils.filesystem import criar_diretorio_de_saida


class PathManager:
    """Gerencia caminhos de entrada e saída."""
    
    def __init__(self, project_root: str, io_config):
        self.project_root = project_root
        self.io_config = io_config
        self.output_dir = None
        
    def setup_paths(self) -> Tuple[str, str, str]:
        """Configura e retorna os caminhos de saída."""
        if self.io_config.CRIAR_DIRETORIO_SAIDA:
            self.output_dir = criar_diretorio_de_saida(self.project_root)
            video_path = os.path.join(self.output_dir, self.io_config.VIDEO_OUTPUT)
            csv_path = os.path.join(self.output_dir, self.io_config.CSV_OUTPUT)
            malha_path = os.path.join(self.output_dir, self.io_config.MALHA_EXPORT_FILENAME)
        else:
            # Mesmo se não criar diretório, baseia os caminhos no raiz do projeto
            video_path = os.path.join(self.project_root, self.io_config.VIDEO_OUTPUT)
            csv_path = os.path.join(self.project_root, self.io_config.CSV_OUTPUT)
            malha_path = os.path.join(self.project_root, self.io_config.MALHA_EXPORT_FILENAME)
        
        return video_path, csv_path, malha_path