import cv2
import numpy as np
from tkinter import messagebox

def run_detection(cv_img, target_class="person", confidence_threshold=0.5):
    """Wykrywa obiekty na obrazie i zwraca obraz z zaznaczonymi detekcjami.
    
    Args:
        cv_img (numpy.ndarray): Obraz w formacie BGR (OpenCV)
        target_class (str): Klasa obiektów do wykrycia (domyślnie 'person')
        confidence_threshold (float): Próg pewności dla detekcji
        
    Returns:
        numpy.ndarray: Obraz z zaznaczonymi detekcjami (BGR)
    """
    try:
        # Ładowanie modelu YOLO
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        height, width = cv_img.shape[:2]
        
        # Przygotowanie obrazu dla sieci
        blob = cv2.dnn.blobFromImage(cv_img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Warstwy wyjściowe
        output_layers_names = net.getUnconnectedOutLayersNames()
        layer_outputs = net.forward(output_layers_names)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold and classes[class_id].lower() == target_class.lower():
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        
        # Rysowanie bounding boxów
        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = {
            "person": (0, 255, 0),    # zielony dla osób
            "car": (255, 0, 0),       # niebieski dla samochodów
            "dog": (0, 0, 255),       # czerwony dla psów
            "cat": (255, 255, 0),     # cyjan dla kotów
        }
        color = colors.get(target_class.lower(), (0, 255, 255))  # żółty dla innych klas
        thickness = 2
        
        # Obsługa różnych formatów zwracanych przez NMSBoxes
        if len(indexes) > 0:
            if isinstance(indexes, (tuple, list)):
                indexes = indexes[0]
            indexes = np.array(indexes).flatten()
            
            for i in indexes:
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(cv_img, label, (x, y - 5), font, 0.5, color, thickness)
        
        return cv_img
    
    except Exception as e:
        messagebox.showerror("Błąd", f"Wystąpił błąd podczas detekcji: {str(e)}")
        return cv_img