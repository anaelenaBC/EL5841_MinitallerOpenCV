import cv2
import os

folder_path = "input"
faces_folder = "caras"
no_faces_folder = "noCaras"

os.makedirs(faces_folder, exist_ok=True)
os.makedirs(no_faces_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            dest_folder = faces_folder
            print(f"{filename}: Se ha detectado una cara")
        else:
            dest_folder = no_faces_folder
            print(f"{filename}: No se ha detectado una cara")
        cv2.imwrite(os.path.join(dest_folder, filename), img)
