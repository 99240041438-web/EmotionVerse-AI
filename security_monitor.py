# =========================================
# EmotionVerse AI - Security Monitor
# =========================================

failed_attempts = 0

print("====================================")
print("   EmotionVerse AI Security Monitor")
print("====================================")

print()
print("Type:")
print("  success  -> successful login")
print("  failed   -> failed login")
print("  exit     -> stop monitoring")
print()


while True:

    event = input("Enter security event: ").lower().strip()


    # -----------------------------------------
    # Successful login
    # -----------------------------------------

    if event == "success":

        print("✅ Successful login")

        # Reset failed attempts
        failed_attempts = 0


    # -----------------------------------------
    # Failed login
    # -----------------------------------------

    elif event == "failed":

        failed_attempts += 1

        print("❌ Failed login")

        print(
            "Failed attempts:",
            failed_attempts
        )


        # Security warning
        if failed_attempts >= 3:

            print()
            print("🚨 SECURITY WARNING!")
            print(
                "Multiple failed login attempts detected."
            )

        else:

            print("No major security event.")


    # -----------------------------------------
    # Exit
    # -----------------------------------------

    elif event == "exit":

        print("Security monitor stopped.")
        break


    # -----------------------------------------
    # Invalid input
    # -----------------------------------------

    else:

        print(
            "Please enter: success, failed, or exit"
        )