
import cv2
import numpy as np

class BackgroundColorExtractor:

    def __init__(self):
        self.background_color = None

    def learn_from_image(self, image_path):

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[BackgroundColorExtractor] Erro: Não foi possível ler a imagem em {image_path}")
                return

            # Calcula a cor média da imagem
            mean_color = np.mean(image, axis=(0, 1))
            self.background_color = tuple(map(int, mean_color))
            print(f"[BackgroundColorExtractor] Cor de fundo aprendida: {self.background_color}")

        except Exception as e:
            print(f"[BackgroundColorExtractor] Erro ao processar a imagem de fundo: {e}")

    def get_background_color(self):
        return self.background_color
