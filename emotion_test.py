from huggingface_hub import hf_hub_download
import tensorflow as tf

print("Downloading emotion model...")

model_path = hf_hub_download(
    repo_id="lokeshkumar79/facial-emotion-recognition",
    filename="finalfacialemotionmodel.keras"
)

print("Model downloaded!")
print("Loading model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully! 🎉")