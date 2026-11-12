
import cv2
import os
import numpy as np
import json

dataset_path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

person_id = 0

for person in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path, person
    )

    if not os.path.isdir(person_folder):
        continue

    label_names[person_id] = person

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(
            person_folder, image_name
        )

        image = cv2.imread(
            image_path, cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(person_id)

    person_id += 1

if len(faces) == 0:
    print("No training images found!")
    exit()

recognizer.train(
    faces, np.array(labels)
)

os.makedirs("model", exist_ok=True)

recognizer.save("model/face_model.yml")

with open("model/labels.json", "w") as file:
    json.dump(label_names, file)

print("Model trained successfully!")
print("Total images:", len(faces))
print("Registered people:", len(label_names))
