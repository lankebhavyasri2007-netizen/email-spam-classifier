
import streamlit as st
import joblib


# -----------------------------
# Load model and vectorizer
# -----------------------------

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("📧 Email Spam Classifier")

st.write(
    "Enter an email below and the machine learning model "
    "will classify it as Spam or Not Spam."
)


# -----------------------------
# Email input
# -----------------------------

email = st.text_area(
    "Enter your email:",
    height=200,
    placeholder="Type or paste an email here..."
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Check Email", use_container_width=True):

    if email.strip() == "":
        st.warning("⚠️ Please enter an email.")

    else:

        # Convert email into TF-IDF features
        email_vector = vectorizer.transform([email])

        # Make prediction
        prediction = model.predict(email_vector)[0]

        # Get probability
        probability = model.predict_proba(email_vector)[0]

        # Display result
        if prediction == 1:

            st.error("🚨 SPAM DETECTED")

            st.write(
                f"Spam probability: **{probability[1] * 100:.2f}%**"
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                f"Not-Spam probability: **{probability[0] * 100:.2f}%**"
            )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Machine Learning Project | Email Spam Classification"
)
