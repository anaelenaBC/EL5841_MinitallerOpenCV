import cv2
import numpy as np

# Inicializar la cámara web (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

# Variables de estado para la calculadora
stage = "num1"        # Etapa actual: num1, operacion, num2, resultado
num1 = None          # Primer número ingresado
num2 = None          # Segundo número ingresado
operacion = None     # Operación matemática (+, -, *, /)
resultado = None     # Resultado del cálculo

def detectar_dedos(frame, x0=50, y0=50, x1=550, y1=550):
    """
    Función principal para detectar el número de dedos levantados en la mano
    Usa análisis de contornos y defectos de convexidad
    """
    # Extraer la región de interés (ROI) donde está la mano
    roi = frame[y0:y1, x0:x1]
    
    # Convertir a escala de grises para procesamiento
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    
    # Aplicar umbralización binaria inversa con método OTSU
    # Esto convierte la imagen a blanco y negro, destacando la mano
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    finger_count = 0  # Contador de dedos detectados
    
    if contours:
        # Seleccionar el contorno más grande (la mano)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        
        # Calcular la envolvente convexa (hull) del contorno
        hull_points = cv2.convexHull(cnt)
        
        # Dibujar el contorno y la envolvente convexa para visualización
        cv2.drawContours(roi, [cnt], -1, (128, 128, 128), 4)
        cv2.drawContours(roi, [hull_points], -1, (128, 128, 128), 4)
        
        # Obtener los índices de la envolvente convexa
        hull = cv2.convexHull(cnt, returnPoints=False)
        
        # Calcular los defectos de convexidad (hundimientos entre dedos)
        defects = cv2.convexityDefects(cnt, hull)
        
        if defects is not None:
            # Analizar cada defecto de convexidad
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]  # start, end, far, distance
                start = tuple(cnt[s][0])    # Punto de inicio del defecto
                end = tuple(cnt[e][0])      # Punto final del defecto
                far = tuple(cnt[f][0])      # Punto más lejano del defecto
                
                # Calcular las distancias entre los puntos
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                
                # Calcular el ángulo usando la ley del coseno
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                
                # Si el ángulo es menor a 90 grados, es un dedo
                if angle <= np.pi/2:
                    finger_count += 1
                    # Dibujar un círculo en el punto del dedo
                    cv2.circle(roi, far, 12, (0,0,0), -1)
    
    return finger_count, thresh

# BUCLE PRINCIPAL: Captura y procesamiento de video en tiempo real
while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:  # Si no se pudo capturar el frame, salir del bucle
        break
    
    # Voltear la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Detectar el número de dedos en el frame actual
    finger_count, thresh = detectar_dedos(frame)
    
    # Definir el área de detección y dibujar un rectángulo alrededor
    x0, y0, x1, y1 = 50, 50, 550, 550
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255,255,255), 2)
    
    # Crear superposición para el texto
    overlay = frame.copy()
    
    # Mostrar el texto de instrucciones según la etapa actual
    if stage == "num1":
        stage_text = "Detecta primer numero (ESPACIO para confirmar)"
    elif stage == "operacion":
        stage_text = "Ingresa operacion (+, -, *, /)"
    elif stage == "num2":
        stage_text = "Detecta segundo numero (ESPACIO para confirmar)"
    elif stage == "resultado":
        stage_text = f"Resultado: {resultado} (ESPACIO para continuar, ENTER para reiniciar)"
    
    # Dibujar fondo negro semi-transparente para el texto de instrucciones
    (text_width, text_height), _ = cv2.getTextSize(stage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(overlay, (10,10), (10 + text_width + 10, 10 + text_height + 10), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, stage_text, (15, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    
    # Mostrar el número actual detectado (dedos + 1)
    current_number_text = f"Numero actual: {finger_count+1}"
    (num_width, num_height), _ = cv2.getTextSize(current_number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(overlay, (x0, y1 + 10), (x0 + num_width + 10, y1 + 10 + num_height + 5), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, current_number_text, (x0 + 5, y1 + 10 + num_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    
    # Mostrar las ventanas
    cv2.imshow("Calculadora con manos", frame)
    cv2.imshow("Threshold", thresh)  # Ventana con la imagen umbralizada para debug
    
    # Capturar teclas presionadas
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break
    # LÓGICA DE CONTROL: Manejar las teclas presionadas
    if key == 32:  # ESPACIO para confirmar
        if stage == "num1":
            # Confirmar el primer número (dedos + 1)
            num1 = finger_count + 1
            stage = "operacion"
        elif stage == "num2":
            # Confirmar el segundo número y realizar el cálculo
            num2 = finger_count + 1
            try:
                # Realizar la operación matemática según el operador
                if operacion == "+":
                    resultado = num1 + num2
                elif operacion == "-":
                    resultado = num1 - num2
                elif operacion == "*":
                    resultado = num1 * num2
                elif operacion == "/":
                    resultado = num1 / num2
                else:
                    resultado = "Operacion invalida"
            except:
                resultado = "Error"  # Manejar errores de división por cero, etc.
            stage = "resultado"
        elif stage == "resultado":
            # Usar el resultado como primer número para continuar calculando
            num1 = resultado
            stage = "operacion"
    
    # Capturar operaciones matemáticas cuando estamos en la etapa "operacion"
    if stage == "operacion":
        if key in [ord('+'), ord('-'), ord('*'), ord('/')]:
            operacion = chr(key)  # Convertir código ASCII a carácter
            stage = "num2"
    
    # ENTER para reiniciar la calculadora
    if key == 13:  # ENTER
        stage = "num1"
        num1 = num2 = resultado = operacion = None

# Limpiar recursos: liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()