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

    if emotion == "Angry":
        risk_score += 20

    elif emotion == "Fear":
        risk_score += 15

    elif emotion == "Sad":
        risk_score += 5

    elif emotion == "Surprise":
        risk_score += 5

    if failed_attempts >= 5:
        risk_score += 50

    elif failed_attempts >= 3:
        risk_score += 30

    elif failed_attempts >= 1:
        risk_score += 10

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

failed_attempts = int(
    input("Enter failed login attempts: ")
)


# =========================================
# 5. OPEN CAMERA
# =========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not access camera.")
    exit()


# =========================================
# 6. VIDEO RECORDER
# =========================================

frame_width = int(
    camera.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = 20.0

video_writer = cv2.VideoWriter(
    "emotion_security_video.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (frame_width, frame_height)
)

print("Video recording started!")
print("Press Q to stop.")


# =========================================
# 7. FACE DETECTOR
# =========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================================
# 8. CAMERA LOOP
# =========================================

while True:

    ret, frame = camera.read()

    if not ret:
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


    # =====================================
    # 9. PROCESS FACE
    # =====================================

    for (x, y, w, h) in faces:

        face = gray[
            y:y + h,
            x:x + w
        ]

        face = cv2.resize(
            face,
            (48, 48)
        )

        face = face / 255.0

        face = np.expand_dims(
            face,
            axis=0
        )

        face = np.expand_dims(
            face,
            axis=-1
        )


        # Emotion prediction
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


        # Risk calculation
        risk_score, risk_level = calculate_risk(
            emotion,
            failed_attempts
        )


        # Draw face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Emotion
        cv2.putText(
            frame,
            f"Emotion: {emotion} ({confidence:.1f}%)",
            (x, y - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # Failed logins
        cv2.putText(
            frame,
            f"Failed Logins: {failed_attempts}",
            (x, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        # Risk
        cv2.putText(
            frame,
            f"Risk: {risk_level} Score: {risk_score}",
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


    # =====================================
    # 10. SAVE FRAME TO VIDEO
    # =====================================

    video_writer.write(frame)


    # =====================================
    # 11. SHOW CAMERA
    # =====================================

    cv2.imshow(
        "EmotionVerse AI - Video Security",
        frame
    )


    # =====================================
    # 12. STOP
    # =====================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================
# 13. CLOSE EVERYTHING
# =========================================

camera.release()

video_writer.release()

cv2.destroyAllWindows()

print()
print("Video saved successfully!")
print("File name: emotion_security_video.avi")