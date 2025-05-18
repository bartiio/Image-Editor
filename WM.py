import cv2
import numpy as np

def Water(obraz_path, znak_path, wspolczynnik=0.5, znak_rozmiar=(20, 20)):
    """
    Funkcja nakłada znak wodny na obraz wielokrotnie. Jeśli znak wodny ma kanał alfa,
    zostanie on odpowiednio uwzględniony.
    """
    # Wczytanie obrazu i znaku wodnego
    obraz = cv2.imread(obraz_path, cv2.IMREAD_COLOR)
    znak = cv2.imread(znak_path, cv2.IMREAD_UNCHANGED)

    if obraz is None or znak is None:
        raise ValueError("Nie udało się wczytać obrazu lub znaku wodnego.")

    # Dopasowanie rozmiaru znaku wodnego (np. 20x20px)
    znak = cv2.resize(znak, znak_rozmiar)

    # Jeśli znak ma kanał alfa
    if znak.shape[2] == 4:
        znak_rgb = znak[:, :, :3]
        alpha = znak[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]  # dopasowanie wymiarów do (20, 20, 1)
    else:
        znak_rgb = znak
        alpha = np.ones((znak.shape[0], znak.shape[1], 1), dtype=np.float32)  # brak przezroczystości

    znak_rgb = znak_rgb.astype(np.float32)
    alpha = alpha.astype(np.float32)
    obraz = obraz.astype(np.float32)

    h_img, w_img = obraz.shape[:2]
    h_znak, w_znak = znak_rgb.shape[:2]

    for y in range(0, h_img, h_znak):
        for x in range(0, w_img, w_znak):
            # Pobierz fragment obrazu
            fragment = obraz[y:y + h_znak, x:x + w_znak]
            h_frag, w_frag = fragment.shape[:2]

            # Dopasuj znak i alpha, jeśli jesteśmy na krawędzi
            znak_crop = znak_rgb[:h_frag, :w_frag]
            alpha_crop = alpha[:h_frag, :w_frag]

            # Nakładanie znaku wodnego
            obraz[y:y + h_frag, x:x + w_frag] = (
                fragment * (1 - alpha_crop * wspolczynnik) +
                znak_crop * alpha_crop * wspolczynnik
            )

    return np.clip(obraz, 0, 255).astype(np.uint8)
