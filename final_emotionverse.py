import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
import threading


# =========================================
# 1. LOAD EMOTION MODEL
# =========================================

print("Loading EmotionVerse AI model...")

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
# 3. SECURITY VARIABLES
# =========================================

failed_attempts = 0

security_running = True


# =========================================
# 4. RISK ENGINE
# =========================================

def calculate_risk(emotion, failed_attempts):

    risk_score = 0

    # Security activity is the main factor
    if failed_attempts >= 5:
        risk_score += 70

    elif failed_attempts >= 3:
        risk_score += 40

    elif failed_attempts >= 1:
        risk_score += 15


    # Emotion is only contextual
    if emotion == "Angry":
        risk_score += 5

    elif emotion == "Fear":
        risk_score += 5

    elif emotion == "Surprise":
        risk_score += 2


    # Determine risk level
    if risk_score >= 60:

        risk_level = "HIGH"

    elif risk_score >= 30:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    return risk_score, risk_level


# =========================================
# 5. SECURITY MONITOR
# =========================================

def security_monitor():

    global failed_attempts
    global security_running

    print()
    print("======================================")
    print("       SECURITY MONITOR")
    print("======================================")
    print()
    print("Commands:")
    print("failed  -> failed login")
    print("success -> successful login")
    print("exit    -> stop security monitor")
    print()


    while security_running:

        event = input(
            "Security event: "
        ).lower().strip()


        # -------------------------------
        # Failed login
        # -------------------------------

        if event == "failed":

            failed_attempts += 1

            print(
                "❌ Failed login detected!"
            )

            print(
                "Failed attempts:",
                failed_attempts
            )


            if failed_attempts >= 3:

                print(
                    "🚨 MULTIPLE FAILED LOGINS!"
                )


        # -------------------------------
        # Successful login
        # -------------------------------

        elif event == "success":

            failed_attempts = 0

            print(
                "✅ Successful login."
            )

            print(
                "Failed attempts reset to 0."
            )


        # -------------------------------
        # Exit
        # -------------------------------

        elif event == "exit":

            security_running = False

            print(
                "Security monitor stopped."
            )


        else:

            print(
                "Use: failed, success, or exit"
            )


# =========================================
# 6. START SECURITY MONITOR
# =========================================

monitor_thread = threading.Thread(
    target=security_monitor
)

monitor_thread.daemon = True

monitor_thread.start()


# =========================================
# 7. OPEN CAMERA
# =========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not access camera.")

    security_running = False

    exit()


# =========================================
# 8. VIDEO RECORDING
# =========================================

frame_width = int(
    camera.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = 20.0


video_writer = cv2.VideoWriter(
    "final_emotionverse_video.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (frame_width, frame_height)
)


print()
print("🎥 Video recording started!")
print("Press Q on the camera window to stop.")
print()


# =========================================
# 9. FACE DETECTOR
# =========================================

face_detector = cv2.CascadeClassifier(

    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# =========================================
# 10. CAMERA LOOP
# =========================================

while True:

    ret, frame = camera.read()


    if not ret:

        print(
            "Could not read camera."
        )

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
    # 11. PROCESS EACH FACE
    # =====================================

    for (x, y, w, h) in faces:


        # Crop face

        face = gray[
            y:y + h,
            x:x + w
        ]


        # Resize

        face = cv2.resize(
            face,
            (48, 48)
        )


        # Normalize

        face = face / 255.0


        # Prepare CNN input

        face = np.expand_dims(
            face,
            axis=0
        )

        face = np.expand_dims(
            face,
            axis=-1
        )


        # =================================
        # 12. EMOTION PREDICTION
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

            np.max(prediction)
            * 100
        )


        # =================================
        # 13. GET CURRENT SECURITY DATA
        # =================================

        current_failed_attempts = (
            failed_attempts
        )


        # =================================
        # 14. RISK CALCULATION
        # =================================

        risk_score, risk_level = (

            calculate_risk(

                emotion,

                current_failed_attempts
            )
        )


        # =================================
        # 15. FACE RECTANGLE
        # =================================

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2
        )


        # =================================
        # 16. EMOTION DISPLAY
        # =================================

        cv2.putText(

            frame,

            f"Emotion: {emotion} "
            f"({confidence:.1f}%)",

            (x, y - 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2
        )


        # =================================
        # 17. SECURITY DISPLAY
        # =================================

        cv2.putText(

            frame,

            f"Failed Logins: "
            f"{current_failed_attempts}",

            (x, y - 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 0),

            2
        )


        # =================================
        # 18. RISK DISPLAY
        # =================================

        cv2.putText(

            frame,

            f"Risk: {risk_level} "
            f"Score: {risk_score}",

            (x, y - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 0, 255),

            2
        )


        # =================================
        # 19. HIGH RISK ALERT
        # =================================

        if risk_level == "HIGH":

            cv2.putText(

                frame,

                "!!! SECURITY ALERT !!!",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                1.0,

                (0, 0, 255),

                3
            )


    # =====================================
    # 20. SAVE VIDEO
    # =====================================

    video_writer.write(frame)


    # =====================================
    # 21. SHOW CAMERA
    # =====================================

    cv2.imshow(

        "EmotionVerse AI - "
        "Cybersecurity Monitor",

        frame
    )


    # =====================================
    # 22. QUIT
    # =====================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================
# 23. CLOSE EVERYTHING
# =========================================

security_running = False

camera.release()

video_writer.release()

cv2.destroyAllWindows()


print()
print("======================================")
print("EmotionVerse AI stopped.")
print("Video saved as:")
print("final_emotionverse_video.avi")
print("===================================