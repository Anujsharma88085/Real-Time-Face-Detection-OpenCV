import cv2
import os

name = input("Enter person's name: ").strip()

if not name:
    print("Please enter a valid name.")
    exit()

folder = os.path.join("dataset", name)
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError(
        "Error: Haar Cascade XML file not found!"
    )

print("Face detector loaded successfully!")

count = 0

print("Press S to save a face image.")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera not available.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray, 1.3, 5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame, (x, y),
            (x+w, y+h), (0, 255, 0), 2
        )

    cv2.putText(
        frame, f"Images: {count}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.imshow("Face Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            count += 1
            path = os.path.join(
                folder, f"{count}.jpg"
            )
            cv2.imwrite(path, face)

        print("Images saved:", count)

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Collection completed.")
