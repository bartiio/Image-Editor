import cv2
import numpy as np

def Water(obraz, znak_rozmiar=(40, 40), wspolczynnik=0.9):
    """
    Dodaje znak wodny w postaci tekstu na obrazie, dopasowując kolor tekstu do tła.

    :param obraz: Obraz wejściowy (PIL.Image lub NumPy array).
    :param znak_rozmiar: Rozmiar znaku wodnego (nieużywane wprost, można dostosować).
    :param wspolczynnik: Przezroczystość (0 = niewidoczny, 1 = pełna widoczność).
    :return: Obraz NumPy z naniesionym znakiem wodnym.
    """
    text = "A K"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    opacity = wspolczynnik

    # Konwersja do NumPy array, jeśli to PIL.Image
    image_array = np.array(obraz)

    # Kopia do rysowania
    overlay = image_array.copy()
    h_img, w_img = image_array.shape[:2]

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_w, text_h = text_size

    for i in range(0, w_img, text_w + 40):
        for j in range(0, h_img, text_h + 40):
            if i + text_w <= w_img and j + text_h <= h_img:
                # Oblicz średnią jasność pod tekstem (w skali szarości)
                roi = image_array[j:j+text_h, i:i+text_w]
                if roi.size == 0:
                    continue
                if len(roi.shape) == 3 and roi.shape[2] == 3:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = roi  # już jest grayscale

                brightness = np.mean(gray_roi)

                # Jasność 0-255: jasne tło → czarny tekst, ciemne tło → biały tekst
                color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

                # Rysuj tekst
                cv2.putText(overlay, text, (i, j + text_h), font, font_scale, color, thickness, cv2.LINE_AA)

    # Łączenie przezroczyste
    cv2.addWeighted(overlay, opacity, image_array, 1 - opacity, 0, image_array)

    return image_array
