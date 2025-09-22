import cv2

# Cargar los clasificadores en cascada de Haar para detección de caras y ojos
# Estos archivos XML contienen los patrones entrenados para detectar caras y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicializar la cámara web (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

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
    # Parámetros: imagen, factor de escala (1.3), vecinos mínimos (5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Contador de usuarios que están mirando a la cámara
    users_looking = 0 

    # Procesar cada cara detectada
    for (x, y, w, h) in faces:
        # Extraer la región de interés (ROI) de la cara en escala de grises y color
        roi_gray = gray[y:y+h, x:x+w]      # ROI en escala de grises para detección de ojos
        roi_color = frame[y:y+h, x:x+w]    # ROI en color para dibujar sobre la cara
        
        # Dibujar un rectángulo blanco alrededor de la cara detectada
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        # Detectar ojos dentro de la región de la cara
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eye_centers = []  # Lista para almacenar los centros de los ojos
        
        # Procesar cada ojo detectado
        for (ex, ey, ew, eh) in eyes:
            # Dibujar un rectángulo gris alrededor de cada ojo
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (128, 128, 128), 2)
            
            # Calcular el centro del ojo
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            eye_centers.append((center_x, center_y))
            
            # Dibujar un círculo rojo en el centro de cada ojo
            cv2.circle(roi_color, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # Si se detectaron al menos 2 ojos, la persona está mirando a la cámara
        if len(eye_centers) >= 2:
            users_looking += 1  # Incrementar contador de usuarios mirando
            
            # Crear superposición semi-transparente para el texto
            overlay = frame.copy()
            text = "Mirando a la camara"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            # Dibujar fondo negro semi-transparente para el texto
            cv2.rectangle(overlay, (x, y-40), (x + text_width + 10, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Mostrar texto "Mirando a la camara" sobre la cara
            cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Mostrar contador global de usuarios mirando a la cámara
    overlay = frame.copy()
    text_users = f"Usuarios mirando: {users_looking}"
    (text_width, text_height), _ = cv2.getTextSize(text_users, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
    # Posición del contador en la esquina superior izquierda
    x_box = 10
    y_box = 10
    
    # Dibujar fondo negro semi-transparente para el contador
    cv2.rectangle(overlay, (x_box-5, y_box), (x_box + text_width + 10, y_box + text_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Mostrar el contador de usuarios mirando
    cv2.putText(frame, text_users, (x_box, y_box + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Mostrar el frame en una ventana
    cv2.imshow("Deteccion ojos", frame)
    
    # Salir del bucle si se presiona la tecla ESC (código 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Limpiar recursos: liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()

