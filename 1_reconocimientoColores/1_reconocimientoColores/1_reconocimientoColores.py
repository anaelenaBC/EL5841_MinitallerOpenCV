import cv2
import numpy as np

# Inicializar la cámara web (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

# Definir el área de interés (ROI) donde se detectará el color
# x, y: coordenadas de la esquina superior izquierda
# w, h: ancho y alto del rectángulo de detección
x, y, w, h = 10, 10, 200, 200

# Diccionario con los rangos HSV para cada color
# Cada color tiene rangos de valores Hue (matiz), Saturation (saturación), Value (brillo)
# El rojo necesita dos rangos porque el rojo está en ambos extremos del espectro HSV
colors_hsv = {
    "Rojo": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
    "Verde": [(36, 50, 70), (89, 255, 255)],
    "Azul": [(90, 50, 70), (128, 255, 255)],
    "Amarillo": [(20, 100, 100), (35, 255, 255)],
    "Naranja": [(10, 100, 20), (25, 255, 255)],
    "Morado": [(129, 50, 70), (158, 255, 255)]
}

def get_dominant_color_hsv(roi):
    # Convertir la imagen de BGR a HSV para mejor detección de colores
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Separar los canales HSV (Hue, Saturation, Value)
    h, s, v = cv2.split(hsv_roi)
    
    # Calcular los valores promedio de saturación y brillo
    mean_v = np.mean(v)  # Brillo promedio
    mean_s = np.mean(s)  # Saturación promedio
    
    # Detectar colores especiales basados en brillo y saturación
    if mean_v < 40:  # Si el brillo es muy bajo, es negro
        return "Negro"
    elif mean_v > 200 and mean_s < 30:  # Si es muy brillante pero poco saturado, es blanco
        return "Blanco"
    
    # Diccionario para almacenar el porcentaje de cada color detectado
    color_percentages = {}
    
    # Iterar sobre cada color definido en el diccionario
    for color, ranges in colors_hsv.items():
        if color == "Rojo":
            # El rojo necesita dos máscaras porque está en ambos extremos del espectro HSV
            lower1, upper1, lower2, upper2 = ranges
            mask1 = cv2.inRange(hsv_roi, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_roi, np.array(lower2), np.array(upper2))
            mask = mask1 + mask2  # Combinar ambas máscaras
        else:
            # Para otros colores, usar un solo rango
            lower, upper = ranges
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        
        # Contar píxeles que coinciden con el rango de color
        color_percentages[color] = cv2.countNonZero(mask)
    
    # Encontrar el color con más píxeles coincidentes (color dominante)
    dominant_color = max(color_percentages, key=color_percentages.get)
    return dominant_color

def color_name_to_bgr(name):
    """
    Convierte el nombre del color a valores BGR para OpenCV
    OpenCV usa el formato BGR (Blue, Green, Red) en lugar de RGB
    """
    mapping = {
        "Rojo": (0,0,255),      # BGR: 0 azul, 0 verde, 255 rojo
        "Verde": (0,255,0),     # BGR: 0 azul, 255 verde, 0 rojo
        "Azul": (255,0,0),      # BGR: 255 azul, 0 verde, 0 rojo
        "Amarillo": (0,255,255), # BGR: 0 azul, 255 verde, 255 rojo
        "Naranja": (0,165,255),  # BGR: 0 azul, 165 verde, 255 rojo
        "Morado": (255,0,255),   # BGR: 255 azul, 0 verde, 255 rojo
        "Blanco": (255,255,255), # BGR: 255 azul, 255 verde, 255 rojo
        "Negro": (0,0,0)         # BGR: 0 azul, 0 verde, 0 rojo
    }
    return mapping.get(name, (255,255,255))  # Color por defecto: blanco

# BUCLE PRINCIPAL: Captura y procesamiento de video en tiempo real
while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:  # Si no se pudo capturar el frame, salir del bucle
        break
    
    # Voltear la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Extraer la región de interés (ROI) donde se detectará el color
    roi = frame[y:y+h, x:x+w]
    
    # Detectar el color dominante en la ROI
    color_name = get_dominant_color_hsv(roi)
    
    # Dibujar un rectángulo blanco alrededor del área de detección
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 3)
    
    # Dibujar un rectángulo pequeño con el color detectado (muestra de color)
    cv2.rectangle(frame, (x+w+10, y), (x+w+60, y+50), color_name_to_bgr(color_name), -1)
    
    # Crear una superposición semi-transparente para el texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y+h+10), (x+250, y+h+60), (0,0,0), -1)
    alpha = 0.6  # Nivel de transparencia (0 = transparente, 1 = opaco)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Mostrar el nombre del color detectado en la pantalla
    cv2.putText(frame, f"Color: {color_name}", (x+5, y+h+45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    # Mostrar el frame en una ventana
    cv2.imshow("Detector de color HSV", frame)
    
    # Salir del bucle si se presiona la tecla ESC (código 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Limpiar recursos: liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()