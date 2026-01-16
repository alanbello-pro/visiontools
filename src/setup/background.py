import os
from typing import Optional
from src.setup.background_color_extractor import BackgroundColorExtractor


class BackgroundSetup:

    @staticmethod
    def setup_background(color_config) -> Optional[list]:
        bg_path = color_config.BACKGROUND_IMAGE_PATH
        
        if not bg_path or not os.path.exists(bg_path):
            print("\n[Aviso] Imagem de fundo não configurada ou não encontrada.")
            return None
        
        bg_extractor = BackgroundColorExtractor()
        bg_extractor.learn_from_image(bg_path)
        bg_color = bg_extractor.get_background_color()
        
        return [bg_color] if bg_color else None