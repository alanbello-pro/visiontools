import cv2
import numpy as np

class FeatureExtractor:

    @staticmethod
    def get_dominant_color(frame, bbox, k, max_iter, epsilon, attempts, ignore_colors=None, color_threshold=50):
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                return None

            roi = frame[y1:y2, x1:x2]
            pixels = roi.reshape((-1, 3))
            pixels = np.float32(pixels)

            if len(pixels) < k:
                if len(pixels) == 0:
                    return None
                mean_color = np.mean(pixels, axis=0)
                return tuple(map(int, mean_color))

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

            if ignore_colors:
                valid_centers = []
                valid_labels = []
                
                for i, center in enumerate(centers):
                    is_background = False
                    for bg_color in ignore_colors:
                        distance = np.linalg.norm(center - np.array(bg_color))
                        if distance < color_threshold:
                            is_background = True
                            break
                    
                    if not is_background:
                        valid_centers.append(center)
                        valid_labels.extend(labels[labels == i])
                
                if not valid_centers:
                    return None

                if not valid_labels:
                     return tuple(map(int, valid_centers[0]))

                labels_array = np.array(valid_labels)
                unique_labels, counts = np.unique(labels_array, return_counts=True)
                
                dominant_label = unique_labels[np.argmax(counts)]
                dominant_color = centers[dominant_label]

            else:
                _, counts = np.unique(labels, return_counts=True)
                dominant_color = centers[np.argmax(counts)]

            return tuple(map(int, dominant_color))

        except (cv2.error, ValueError, IndexError) as e:
            print(f"[FeatureExtractor] Erro ao extrair cor dominante para bbox {bbox}: {e}")
            return None

    @staticmethod
    def calculate_hs_histogram(frame, bbox, bins=(180, 256)):

        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)

            # Clipping: Garante que a bounding box esteja dentro dos limites do frame.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                return None

            roi = frame[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            hist = cv2.calcHist([hsv_roi], [0, 1], None, bins, [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            
            return hist

        except (cv2.error, ValueError, IndexError) as e:
            print(f"[FeatureExtractor] Erro ao calcular histograma para bbox {bbox}: {e}")
            return None

    @staticmethod
    def compare_histograms(hist1, hist2):
        if hist1 is None or hist2 is None:
            return 1.0

        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        correlation = max(correlation, 0)

        return 1 - correlation
