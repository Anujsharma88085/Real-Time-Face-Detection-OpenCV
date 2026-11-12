
import cv2
import json

# Load trained recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/face_model.yml")

# Load registered names
with open("model/labels.json", "r") as file:
    names = json.load(file)

# Load face detector
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError(
        "Error: Haar Cascade XML file not found!"
    )

print("Face detector loaded successfully!")

# Start webcam
camera = cv2.VideoCapture(0)

print("Face Recognition Started")
print("Press Q to exit")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera not available")
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    detected_faces = face_detector.detectMultiScale(
        gray, 1.3, 5
    )

    for (x, y, w, h) in detected_faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        # Lower confidence means a closer match
        if confidence < 75:
            name = names.get(str(label), "Unknown")
        else:
            name = "Unknown"

        cv2.rectangle(
            frame, (x, y),
            (x+w, y+h), (0, 255, 0), 2
        )

        cv2.putText(
            frame, name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2
        )

    cv2.imshow(
        "Real-Time Face Recognition", frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
