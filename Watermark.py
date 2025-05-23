import cv2
import numpy as np

def Water(obraz, znak_rozmiar=(40, 40), wspolczynnik=0.9, tekst="A K"):
    """
    Dodaje znak wodny w postaci tekstu na obrazie, dopasowując kolor tekstu do tła.
    :param obraz: Obraz wejściowy (PIL.Image lub NumPy array).
    :param znak_rozmiar: Rozmiar znaku wodnego (odstępy).
    :param wspolczynnik: Przezroczystość (0 = niewidoczny, 1 = pełna widoczność).
    :param tekst: Treść znaku wodnego.
    :return: Obraz NumPy z naniesionym znakiem wodnym.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    opacity = wspolczynnik

    image_array = np.array(obraz)
    overlay = image_array.copy()
    h_img, w_img = image_array.shape[:2]

    text_size = cv2.getTextSize(tekst, font, font_scale, thickness)[0]
    text_w, text_h = text_size

    for i in range(0, w_img, text_w + znak_rozmiar[0]):
        for j in range(0, h_img, text_h + znak_rozmiar[1]):
            if i + text_w <= w_img and j + text_h <= h_img:
                roi = image_array[j:j+text_h, i:i+text_w]
                if roi.size == 0:
                    continue
                if len(roi.shape) == 3 and roi.shape[2] == 3:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = roi
                brightness = np.mean(gray_roi)
                color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                cv2.putText(overlay, tekst, (i, j + text_h), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, opacity, image_array, 1 - opacity, 0, image_array)
    return image_array
