import cv2
import numpy as np

cap = cv2.VideoCapture(0)
x, y, w, h = 10, 10, 200, 200

colors_hsv = {
    "Rojo": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
    "Verde": [(36, 50, 70), (89, 255, 255)],
    "Azul": [(90, 50, 70), (128, 255, 255)],
    "Amarillo": [(20, 100, 100), (35, 255, 255)],
    "Naranja": [(10, 100, 20), (25, 255, 255)],
    "Morado": [(129, 50, 70), (158, 255, 255)]
}

def get_dominant_color_hsv(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_roi)
    mean_v = np.mean(v)
    mean_s = np.mean(s)
    if mean_v < 40:
        return "Negro"
    elif mean_v > 200 and mean_s < 30:
        return "Blanco"
    color_percentages = {}
    for color, ranges in colors_hsv.items():
        if color == "Rojo":
            lower1, upper1, lower2, upper2 = ranges
            mask1 = cv2.inRange(hsv_roi, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_roi, np.array(lower2), np.array(upper2))
            mask = mask1 + mask2
        else:
            lower, upper = ranges
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        color_percentages[color] = cv2.countNonZero(mask)
    dominant_color = max(color_percentages, key=color_percentages.get)
    return dominant_color

def color_name_to_bgr(name):
    mapping = {
        "Rojo": (0,0,255),
        "Verde": (0,255,0),
        "Azul": (255,0,0),
        "Amarillo": (0,255,255),
        "Naranja": (0,165,255),
        "Morado": (255,0,255),
        "Blanco": (255,255,255),
        "Negro": (0,0,0)
    }
    return mapping.get(name, (255,255,255))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    roi = frame[y:y+h, x:x+w]
    color_name = get_dominant_color_hsv(roi)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 3)
    cv2.rectangle(frame, (x+w+10, y), (x+w+60, y+50), color_name_to_bgr(color_name), -1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y+h+10), (x+250, y+h+60), (0,0,0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, f"Color: {color_name}", (x+5, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Detector de color HSV", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()