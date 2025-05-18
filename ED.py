import cv2
import numpy as np
from PIL import Image

def ED_canny(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    szary = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    krawedzie = cv2.Canny(szary, 100, 200)
    return Image.fromarray(krawedzie)

def ED_sobel(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    szary = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(szary, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(szary, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)
    return Image.fromarray(sobel)

def ED_laplacian(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    szary = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(szary, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return Image.fromarray(laplacian)
