import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

# -----------------------------------------
# 1. Load the emotion AI model
# -----------------------------------------

print("Loading emotion model...")

model_path = hf_hub_download(
    repo_id="lokeshkumar79/facial-emotion-recognition",
    filename="finalfacialemotionmodel.keras"
)

emotion_model = tf.keras.models.load_model(model_path)

print("Emotion model loaded successfully! 🎉")


# -----------------------------------------
# 2. Emotion names
# -----------------------------------------

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# -----------------------------------------
# 3. Open the laptop camera
# -----------------------------------------

camera = cv2.VideoCapture(0)


# -----------------------------------------
# 4. Load face detector
# -----------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

print("Face detector loaded:", not face_detector.empty())


# -----------------------------------------
# 5. Camera loop
# -----------------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    # Convert camera image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # -----------------------------------------
    # 6. Process every detected face
    # -----------------------------------------

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Crop the face
        face = gray[y:y + h, x:x + w]

        # Resize to 48 x 48
        face = cv2.resize(face, (48, 48))

        # Normalize pixel values
        face = face / 255.0

        # Prepare image for CNN
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # -----------------------------------------
        # 7. Predict emotion
        # -----------------------------------------

        prediction = emotion_model.predict(
            face,
            verbose=0
        )

        emotion_index = np.argmax(prediction)

        emotion = emotion_labels[emotion_index]

        confidence = np.max(prediction) * 100

        # -----------------------------------------
        # 8. Display emotion
        # -----------------------------------------

        text = f"{emotion} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show camera
    cv2.imshow(
        "EmotionVerse AI - Emotion Detection",
        frame
    )

    # Press Q to close
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------------------
# 9. Close everything
# -----------------------------------------

camera.release()
cv2.destroyAllWindows()