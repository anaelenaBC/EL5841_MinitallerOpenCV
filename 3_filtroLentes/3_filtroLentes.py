import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

glasses = cv2.imread("lentes.png", cv2.IMREAD_UNCHANGED)
if glasses is None:
    print("Error: No se pudo cargar la imagen de lentes. Revisa la ruta.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]
            x1 = max(min(ex1, ex2) - 30, 0)
            x2 = min(max(ex1+ew1, ex2+ew2) + 30, w)
            y1 = max(min(ey1, ey2) + 20, 0)
            y2 = min(max(ey1+eh1, ey2+eh2) + 30, h)
            glasses_width = x2 - x1
            glasses_height = int(glasses.shape[0] * (glasses_width / glasses.shape[1]))
            resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height))
            y_offset = y1
            x_offset = x1
            for c in range(0,3):
                alpha = resized_glasses[:,:,3] / 255.0
                for i in range(glasses_height):
                    for j in range(glasses_width):
                        if 0 <= y_offset+i < h and 0 <= x_offset+j < w:
                            roi_color[y_offset+i, x_offset+j, c] = \
                                alpha[i,j]*resized_glasses[i,j,c] + \
                                (1-alpha[i,j])*roi_color[y_offset+i, x_offset+j, c]
    cv2.imshow("Filtro de gafas - OpenCV", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()