import cv2

# Cargar los clasificadores en cascada de Haar para detección de caras y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicializar la cámara web (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

# Cargar la imagen de lentes con canal alfa (transparencia)
# IMREAD_UNCHANGED preserva el canal alfa para efectos de transparencia
glasses = cv2.imread("lentes.png", cv2.IMREAD_UNCHANGED)
if glasses is None:
    print("Error: No se pudo cargar la imagen de lentes. Revisa la ruta.")
    exit()

# BUCLE PRINCIPAL: Captura y procesamiento de video en tiempo real
while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:  # Si no se pudo capturar el frame, salir del bucle
        break
    
    # Voltear la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Convertir la imagen a escala de grises (necesario para la detección de Haar)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar caras en la imagen usando el clasificador de Haar
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Procesar cada cara detectada
    for (x, y, w, h) in faces:
        # Extraer la región de interés (ROI) de la cara
        roi_gray = gray[y:y+h, x:x+w]      # ROI en escala de grises para detección de ojos
        roi_color = frame[y:y+h, x:x+w]    # ROI en color para aplicar el filtro
        
        # Detectar ojos dentro de la región de la cara
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Solo aplicar lentes si se detectan al menos 2 ojos
        if len(eyes) >= 2:
            # Obtener coordenadas de los dos primeros ojos detectados
            ex1, ey1, ew1, eh1 = eyes[0]  # Primer ojo: x, y, ancho, alto
            ex2, ey2, ew2, eh2 = eyes[1]  # Segundo ojo: x, y, ancho, alto
            
            # Calcular el área donde se colocarán las lentes
            # x1, x2: límites horizontales (con margen de 30 píxeles)
            x1 = max(min(ex1, ex2) - 30, 0)  # Límite izquierdo
            x2 = min(max(ex1+ew1, ex2+ew2) + 30, w)  # Límite derecho
            
            # y1, y2: límites verticales (con margen de 20-30 píxeles)
            y1 = max(min(ey1, ey2) + 20, 0)  # Límite superior
            y2 = min(max(ey1+eh1, ey2+eh2) + 30, h)  # Límite inferior
            
            # Calcular el ancho y alto de las lentes
            glasses_width = x2 - x1
            glasses_height = int(glasses.shape[0] * (glasses_width / glasses.shape[1]))
            
            # Redimensionar la imagen de lentes para que coincida con el área calculada
            resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height))
            
            # Definir la posición donde se colocarán las lentes
            y_offset = y1  # Desplazamiento vertical
            x_offset = x1  # Desplazamiento horizontal
            
            # APLICAR TRANSPARENCIA: Mezclar la imagen de lentes con la cara
            # Procesar cada canal de color (BGR)
            for c in range(0, 3):
                # Obtener el canal alfa (transparencia) normalizado entre 0 y 1
                alpha = resized_glasses[:,:,3] / 255.0
                
                # Aplicar la mezcla píxel por píxel
                for i in range(glasses_height):
                    for j in range(glasses_width):
                        # Verificar que las coordenadas estén dentro de los límites
                        if 0 <= y_offset+i < h and 0 <= x_offset+j < w:
                            # Fórmula de mezcla alfa: resultado = alpha*lentes + (1-alpha)*fondo
                            roi_color[y_offset+i, x_offset+j, c] = \
                                alpha[i,j]*resized_glasses[i,j,c] + \
                                (1-alpha[i,j])*roi_color[y_offset+i, x_offset+j, c]
    # Mostrar el frame con el filtro de lentes aplicado
    cv2.imshow("Filtro de gafas - OpenCV", frame)
    
    # Salir del bucle si se presiona la tecla ESC (código 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Limpiar recursos: liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()