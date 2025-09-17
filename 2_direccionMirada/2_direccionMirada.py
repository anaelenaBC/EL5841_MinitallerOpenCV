import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    users_looking = 0 

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (128, 128, 128), 2)
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            eye_centers.append((center_x, center_y))
            cv2.circle(roi_color, (center_x, center_y), 4, (0, 0, 255), -1)
        if len(eye_centers) >= 2:
            users_looking += 1
            overlay = frame.copy()
            text = "Mirando a la camara"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(overlay, (x, y-40), (x + text_width + 10, y), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    overlay = frame.copy()
    text_users = f"Usuarios mirando: {users_looking}"
    (text_width, text_height), _ = cv2.getTextSize(text_users, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x_box = 10
    y_box = 10
    cv2.rectangle(overlay, (x_box-5, y_box), (x_box + text_width + 10, y_box + text_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text_users, (x_box, y_box + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Deteccion ojos", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

