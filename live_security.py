import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download


# =========================================
# 1. LOAD EMOTION MODEL
# =========================================

print("Loading emotion model...")

model_path = hf_hub_download(
    repo_id="lokeshkumar79/facial-emotion-recognition",
    filename="finalfacialemotionmodel.keras"
)

emotion_model = tf.keras.models.load_model(model_path)

print("Emotion model loaded successfully!")


# =========================================
# 2. EMOTION LABELS
# =========================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# =========================================
# 3. RISK ENGINE
# =========================================

def calculate_risk(emotion, failed_attempts):

    risk_score = 0

    # Emotion signal
    if emotion == "Angry":
        risk_score += 20

    elif emotion == "Fear":
        risk_score += 15

    elif emotion == "Sad":
        risk_score += 5

    elif emotion == "Surprise":
        risk_score += 5

    # Cybersecurity signal
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


# =========================================
# 4. SECURITY EVENT
# =========================================

print()
print("Enter the number of failed login attempts.")

failed_attempts = int(
    input("Failed login attempts: ")
)

print()
print("Security event recorded:", failed_attempts)


# =========================================
# 5. OPEN CAMERA
# =========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not access camera.")
    exit()


# =========================================
# 6. LOAD FACE DETECTOR
# =========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

print(
    "Face detector loaded:",
    not face_detector.empty()
)


# =========================================
# 7. CAMERA LOOP
# =========================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("Could not read camera.")
        break


    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )


    # =====================================
    # 8. PROCESS EACH FACE
    # =====================================

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[
            y:y + h,
            x:x + w
        ]


        # Resize to CNN input size
        face = cv2.resize(
            face,
            (48, 48)
        )


        # Normalize
        face = face / 255.0


        # Add dimensions
        face = np.expand_dims(
            face,
            axis=0
        )

        face = np.expand_dims(
            face,
            axis=-1
        )


        # =================================
        # 9. EMOTION PREDICTION
        # =================================

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


        # =================================
        # 10. RISK CALCULATION
        # =================================

        risk_score, risk_level = calculate_risk(
            emotion,
            failed_attempts
        )


        # =================================
        # 11. DRAW FACE RECTANGLE
        # =================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # =================================
        # 12. DISPLAY EMOTION
        # =================================

        emotion_text = (
            f"Emotion: {emotion} "
            f"({confidence:.1f}%)"
        )

        cv2.putText(
            frame,
            emotion_text,
            (x, y - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # =================================
        # 13. DISPLAY SECURITY EVENT
        # =================================

        login_text = (
            f"Failed Logins: {failed_attempts}"
        )

        cv2.putText(
            frame,
            login_text,
            (x, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )


        # =================================
        # 14. DISPLAY RISK
        # =================================

        risk_text = (
            f"Risk: {risk_level} "
            f"Score: {risk_score}"
        )

        cv2.putText(
            frame,
            risk_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # =====================================
    # 15. SHOW CAMERA
    # =====================================

    cv2.imshow(
        "EmotionVerse AI - Security Monitor",
        frame
    )


    # =====================================
    # 16. PRESS Q TO EXIT
    # =====================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================
# 17. CLOSE CAMERA
# =========================================

camera.release()

cv2.destroyAllWindows()

print("System closed.")