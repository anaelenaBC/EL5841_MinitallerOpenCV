import cv2
import os

# Definir las rutas de las carpetas
folder_path = "input"          # Carpeta con las imágenes de entrada
faces_folder = "caras"         # Carpeta donde se guardarán las imágenes CON caras
no_faces_folder = "noCaras"    # Carpeta donde se guardarán las imágenes SIN caras

# Crear las carpetas de destino si no existen
# exist_ok=True evita errores si las carpetas ya existen
os.makedirs(faces_folder, exist_ok=True)
os.makedirs(no_faces_folder, exist_ok=True)

# Cargar el clasificador en cascada de Haar para detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# BUCLE PRINCIPAL: Procesar cada imagen en la carpeta de entrada
for filename in os.listdir(folder_path):
    # Filtrar solo archivos de imagen (PNG, JPG, JPEG)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construir la ruta completa de la imagen
        img_path = os.path.join(folder_path, filename)
        
        # Cargar la imagen desde el archivo
        img = cv2.imread(img_path)
        
        # Convertir la imagen a escala de grises (necesario para la detección de Haar)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras en la imagen usando el clasificador de Haar
        # scaleFactor=1.3: Factor de escala para la pirámide de imágenes
        # minNeighbors=5: Número mínimo de vecinos para confirmar una detección
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Decidir en qué carpeta guardar la imagen basándose en si tiene caras
        if len(faces) > 0:
            dest_folder = faces_folder  # Carpeta para imágenes CON caras
            print(f"{filename}: Se ha detectado una cara")
        else:
            dest_folder = no_faces_folder  # Carpeta para imágenes SIN caras
            print(f"{filename}: No se ha detectado una cara")
        
        # Guardar la imagen en la carpeta correspondiente
        cv2.imwrite(os.path.join(dest_folder, filename), img)
