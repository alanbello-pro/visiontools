import cv2
from src.setup.config import AnnotationConfig
from src.utils.image_utils import draw_timer_on_frame

class FrameAnnotator:

    def __init__(self, annotator_config: AnnotationConfig):

        self.config = annotator_config
        self.class_colors = self.config.CLASS_COLORS
        self.default_color = (0, 0, 0)
        self.speed_size_cache = {}

    def desenhar_track(self, frame, track_id, classe, box, velocidade):
        x1, y1, x2, y2 = map(int, box)
        
        frame = self._desenhar_deteccao(frame, track_id, classe, x1, y1, x2, y2)

        if velocidade is not None:
            ponto_velocidade = ((x1 + x2) / 2, y2)
            frame = self._desenhar_velocidade(frame, ponto_velocidade, velocidade)
            
        return frame

    def _desenhar_deteccao(self, frame, track_id, classe, x1, y1, x2, y2):
        cor = self._get_class_color(classe)
        label = f"ID: {track_id} {classe}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, self.config.DETECTION_BOX_THICKNESS_PX)
        cv2.putText(frame, label, (x1, y1 - self.config.LABEL_Y_OFFSET_PX),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.ID_FONT_SCALE, cor, self.config.FONT_THICKNESS)
        return frame

    def _get_class_color(self, classe):
        return self.class_colors.get(classe, self.default_color)

    def _desenhar_velocidade(self, frame, posicao, velocidade_kmh):
        x, y = int(posicao[0]), int(posicao[1])
        texto = f"{velocidade_kmh:.1f} km/h"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.FONT_SCALE
        font_thickness = self.config.FONT_THICKNESS
        
        if velocidade_kmh < self.config.SPEED_THRESH_LOW:
            cor = self.config.SPEED_COLOR_LOW
        elif velocidade_kmh < self.config.SPEED_THRESH_MED:
            cor = self.config.SPEED_COLOR_MED
        else:
            cor = self.config.SPEED_COLOR_HIGH
        
        if texto not in self.speed_size_cache:
            self.speed_size_cache[texto] = cv2.getTextSize(texto, font, font_scale, font_thickness)[0]
        
        (text_width, text_height) = self.speed_size_cache[texto]
        text_y = y - self.config.SPEED_ANNOTATION_Y_OFFSET_PX
        text_x = x - text_width // 2
        
        cv2.rectangle(frame, (text_x - self.config.SPEED_ANNOTATION_PADDING_PX, text_y - text_height - self.config.SPEED_ANNOTATION_PADDING_PX), 
                      (text_x + text_width + self.config.SPEED_ANNOTATION_PADDING_PX, text_y + self.config.SPEED_ANNOTATION_PADDING_PX), 
                      self.config.SPEED_ANNOTATION_BG_COLOR_BGR, -1)
        
        cv2.putText(frame, texto, (text_x, text_y), font, font_scale, cor, font_thickness, cv2.LINE_AA)
        
        return frame

    def draw_video_timer(self, im_frame, frame_count, fps):
        return draw_timer_on_frame(
            frame=im_frame,
            frame_count=frame_count,
            fps=fps,
            font_scale=self.config.TIMER_FONT_SCALE,
            font_thickness=self.config.TIMER_FONT_THICKNESS,
            text_color_bgr=tuple(self.config.TIMER_TEXT_COLOR_BGR),
            bg_color_bgr=tuple(self.config.TIMER_BG_COLOR_BGR),
            margin_px=self.config.TIMER_MARGIN_PX,
            padding_px=self.config.TIMER_PADDING_PX
        )