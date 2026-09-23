import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download


# -----------------------------------------
# 1. Load Emotion AI Model
# -----------------------------------------

print("Loading emotion model...")

model_path = hf_hub_download(
    repo_id="lokeshkumar79/facial-emotion-recognition",
    filename="finalfacialemotionmodel.keras"
)

emotion_model = tf.keras.models.load_model(model_path)

print("Emotion model loaded successfully!")


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
# 3. Risk Engine
# -----------------------------------------

def calculate_risk(emotion, failed_attempts):

    risk_score = 0

    # Emotion contribution
    if emotion == "Angry":
        risk_score += 20

    elif emotion == "Fear":
        risk_score += 15

    elif emotion == "Sad":
        risk_score += 5

    elif emotion == "Surprise":
        risk_score += 5

    # Security activity contribution
    if failed_attempts >= 5:
        risk_score += 50

    elif failed_attempts >= 3:
        risk_score += 30

    elif failed_attempts >= 1:
        risk_score += 10

    # Risk level
    if risk_score >= 60:
        risk_level = "HIGH"

    elif risk_score >= 30:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return risk_score, risk_level


# -----------------------------------------
# 4. Simulated security event
# -----------------------------------------

# For now, we are pretending that
# there were 5 failed login attempts.

failed_attempts = 5


# -----------------------------------------
# 5. Open Camera
# -----------------------------------------

camera = cv2.VideoCapture(0)


# -----------------------------------------
# 6. Face Detector
# -----------------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

print("Face detector loaded:",
      not face_detector.empty())


# -----------------------------------------
# 7. Camera Loop
# -----------------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    # -----------------------------------------
    # 8. Process detected faces
    # -----------------------------------------

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face / 255.0

        # Prepare for CNN
        face = np.expand_dims(
            face,
            axis=0
        )

        face = np.expand_dims(
            face,
            axis=-1
        )


        # -----------------------------------------
        # 9. Predict Emotion
        # -----------------------------------------

        prediction = emotion_model.predict(
            face,
            verbose=0
        )

        emotion_index = np.argmax(
            prediction
        )

        emotion = emotion_labels[
            emotion_index
        ]

        confidence = (
            np.max(prediction) * 100
        )


        # -----------------------------------------
        # 10. Calculate Risk
        # -----------------------------------------

        risk_score, risk_level = calculate_risk(
            emotion,
            failed_attempts
        )


        # -----------------------------------------
        # 11. Draw Face
        # -----------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


        # -----------------------------------------
        # 12. Display Emotion
        # -----------------------------------------

        emotion_text = (
            f"Emotion: {emotion} "
            f"({confidence:.1f}%)"
        )

        cv2.putText(
            frame,
            emotion_text,
            (x, y-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # -----------------------------------------
        # 13. Display Risk
        # -----------------------------------------

        risk_text = (
            f"Risk: {risk_level} "
            f"({risk_score})"
        )

        cv2.putText(
            frame,
            risk_text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # -----------------------------------------
    # 14. Display Camera
    # -----------------------------------------

    cv2.imshow(
        "EmotionVerse AI - Security System",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------------------
# 15. Close
# -----------------------------------------

camera.release()

cv2.destroyAllWindows()