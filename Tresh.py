import cv2
import numpy as np
from PIL import Image

def th_const(pil_img, threshold_value=127, max_value=255):
    """
    Wykonuje progowanie z użyciem stałej wartości progowej.
    :param pil_img: Obraz PIL.Image
    :return: Obiekt PIL.Image po progowaniu
    """
    obraz = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    _, progowany = cv2.threshold(szary, threshold_value, max_value, cv2.THRESH_BINARY)
    return Image.fromarray(progowany)

def th_adapt(pil_img, max_value=255):
    """
    Wykonuje progowanie adaptacyjne.
    :param pil_img: Obraz PIL.Image
    :return: Obiekt PIL.Image po progowaniu
    """
    obraz = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    progowany = cv2.adaptiveThreshold(szary, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(progowany)

def th_otsu(pil_img):
    """
    Wykonuje progowanie Otsu.
    :param pil_img: Obraz PIL.Image
    :return: Obiekt PIL.Image po progowaniu
    """
    obraz = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    szary = cv2.cvtColor(obraz, cv2.COLOR_BGR2GRAY)
    _, progowany = cv2.threshold(szary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(progowany)
