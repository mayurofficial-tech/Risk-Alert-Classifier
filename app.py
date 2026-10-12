
import json
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Risk Alert Classifier",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Risk Alert Classifier")
st.write(
    "Enter customer details to estimate the probability of High Risk."
)

MODEL_PATH = "model/model.joblib"
SCHEMA_PATH = "outputs/input_schema.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

try:
    model = load_model()
    schema = load_schema()
except Exception as error:
    st.error(f"Could not load model or schema: {error}")
    st.stop()

st.subheader("Customer Information")

input_values = {}

with st.form("risk_prediction_form"):

    columns = st.columns(2)

    for index, (feature, config) in enumerate(schema.items()):

        with columns[index % 2]:

            if config["type"] == "numeric":

                min_value = float(config["min"])
                max_value = float(config["max"])
                default_value = float(config["default"])

                # Ensure valid input bounds
                if max_value <= min_value:
                    min_value = default_value - 1
                    max_value = default_value + 1

                default_value = min(
                    max(default_value, min_value),
                    max_value
                )

                input_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value
                )

            else:
                choices = config.get("choices", [])

                if not choices:
                    choices = [config.get("default", "Unknown")]

                default = config.get("default", choices[0])

                if default not in choices:
                    choices.insert(0, default)

                input_values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=choices,
                    index=choices.index(default)
                )

    submitted = st.form_submit_button(
        "Predict Risk",
        type="primary"
    )

if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=list(schema.keys())
    )

    try:
        probabilities = model.predict_proba(input_df)[0]
        high_risk_probability = float(probabilities[1])

        predicted_class = int(
            high_risk_probability >= 0.5
        )

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "High Risk Probability",
                f"{high_risk_probability:.2%}"
            )

        with col2:
            st.metric(
                "Low Risk Probability",
                f"{probabilities[0]:.2%}"
            )

        if predicted_class == 1:
            st.error("Predicted Risk Status: HIGH RISK")
        else:
            st.success("Predicted Risk Status: LOW RISK")

        st.caption(
            "This prediction uses a 0.5 probability threshold. "
            "Validate the model and threshold before making real "
            "credit or financial decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
