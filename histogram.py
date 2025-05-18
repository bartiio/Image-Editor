import matplotlib
matplotlib.use("TkAgg")

import cv2
import matplotlib.pyplot as plt
class HistogramViewer:
    def __init__(self, image_path: str = None):
        self.image_path = image_path
        self.image = None
        if image_path:
            self.load_image(image_path)

    def load_image(self, path: str):
        """Wczytuje obraz z podanej ścieżki"""
        self.image_path = path
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"Nie udało się załadować obrazu z: {path}")

    def show_histograms(self):
        """Wyświetla histogramy: czarno-biały i kolorowy"""
        if self.image is None:
            raise RuntimeError("Brak załadowanego obrazu. Użyj najpierw metody `load_image()`.")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

        colors = ('b', 'g', 'r')
        hist_color = {}
        for i, col in enumerate(colors):
            hist_color[col] = cv2.calcHist([self.image], [i], None, [256], [0, 256])

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(hist_gray, color='k')
        plt.title("Histogram - Obraz czarno-biały")
        plt.xlabel("Wartość piksela")
        plt.ylabel("Liczba pikseli")

        plt.subplot(1, 2, 2)
        for col in colors:
            plt.plot(hist_color[col], color=col)
        plt.title("Histogram - Obraz kolorowy")
        plt.xlabel("Wartość piksela")
        plt.ylabel("Liczba pikseli")

        plt.tight_layout()
        plt.show()
