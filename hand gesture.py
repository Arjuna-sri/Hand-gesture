import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

def recognize_gesture(finger_count):
    if finger_count == 0:
        return "Fist"
    elif finger_count == 1:
        return "One"
    elif finger_count == 2:
        return "Peace"
    elif finger_count == 3:
        return "Three"
    elif finger_count == 4:
        return "Four"
    elif finger_count == 5:
        return "Open Hand"
    else:
        return "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  

    
    x_start, y_start, width, height = 50, 50, 400, 400
    roi = frame[y_start:y_start+height, x_start:x_start+width]

    cv2.rectangle(frame, (x_start, y_start), (x_start+width, y_start+height), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(max_contour) > 1000:
            cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 2)

            hull = cv2.convexHull(max_contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(max_contour, hull)
                if defects is not None:
                    count_defects = 0

                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        a = math.dist(start, end)
                        b = math.dist(start, far)
                        c = math.dist(end, far)
                        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c + 1e-5))

                        if angle <= math.pi / 2:
                            count_defects += 1
                            cv2.circle(roi, far, 5, (0, 0, 255), -1)

                    finger_count = count_defects + 1
                    gesture = recognize_gesture(finger_count)

                    cv2.putText(frame, f'{gesture}', (50, 480), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Thresh", thresh)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
