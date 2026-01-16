import cv2

class ResourceManager:
  
    def __init__(self):
        self.video_input = None  
        self.output_buffer = None
        self.csv_adapter = None
        self.video_adapter = None
        
    def cleanup(self, app_config=None):
        
        if self.output_buffer is not None:
            try:
                self.output_buffer.close()
            except Exception as e:
                print(f"Erro ao fechar OutputBuffer: {e}")
        if self.csv_adapter is not None:
            try:
                self.csv_adapter.close()
            except Exception as e:
                print(f"Erro ao fechar CSV adapter: {e}")
        
        if self.video_adapter is not None:
            try:
                self.video_adapter.close()
            except Exception as e:
                print(f"Erro ao fechar vídeo adapter: {e}")
        
        # Só tenta fechar janelas se elas foram abertas
        if app_config and app_config.GENERAL_CONFIG.SHOW_VIDEO_WINDOW:
            cv2.destroyAllWindows()