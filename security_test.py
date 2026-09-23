# -----------------------------------------
# EmotionVerse AI - Security Event Test
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
# Get security information
# -----------------------------------------

emotion = input("Enter emotion: ")
failed_attempts = int(
    input("Enter failed login attempts: ")
)


# -----------------------------------------
# Calculate risk
# -----------------------------------------

score, risk = calculate_risk(
    emotion,
    failed_attempts
)


# -----------------------------------------
# Display result
# -----------------------------------------

print()
print("========== SECURITY RESULT ==========")
print("Emotion:", emotion)
print("Failed attempts:", failed_attempts)
print("Risk Score:", score)
print("Risk Level:", risk)
print("=====================================")