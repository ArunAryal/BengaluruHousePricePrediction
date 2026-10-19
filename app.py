import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(
    os.environ.get("HOUSE_MODEL_PATH", Path(__file__).parent / "house_price_model.joblib")
)
MIN_SQFT_PER_BHK = 300  # the training data was filtered with this rule
DEFAULT_LOCATION = "Whitefield"

st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return joblib.load(path)


def get_locations(model) -> list[str]:
    """Location categories the pipeline's one-hot encoder was fitted on."""
    categories = list(model[0].named_transformers_["cat"].categories_[0])
    named = sorted((c for c in categories if c != "other"), key=str.lower)
    return named + (["other"] if "other" in categories else [])


def location_label(name: str) -> str:
    return "Other (area not listed)" if name == "other" else name


def format_price(lakhs: float) -> str:
    if lakhs >= 100:
        return f"₹ {lakhs / 100:,.2f} crore"
    return f"₹ {lakhs:,.2f} lakh"


st.title("🏠 Bengaluru House Price Predictor")
st.caption("Estimate the price of a flat or house from its location, size and number of rooms.")

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH}`. Save your fitted pipeline from the "
        "notebook with `joblib.dump(pipe, 'house_price_model.joblib')` and place it "
        "next to `app.py`."
    )
    st.stop()

model = load_model(str(MODEL_PATH))
locations = get_locations(model)

with st.form("inputs"):
    location = st.selectbox(
        "Location",
        locations,
        index=locations.index(DEFAULT_LOCATION) if DEFAULT_LOCATION in locations else 0,
        format_func=location_label,
    )
    col_sqft, col_bhk, col_bath = st.columns(3)
    sqft = col_sqft.number_input(
        "Total area (sqft)", min_value=300, max_value=30000, value=1200, step=50
    )
    bhk = col_bhk.number_input("Bedrooms (BHK)", min_value=1, max_value=16, value=2, step=1)
    bath = col_bath.number_input("Bathrooms", min_value=1, max_value=16, value=2, step=1)
    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    row = {
        "location": location,
        "total_sqft": float(sqft),
        "bath": float(bath),
        "bhk": float(bhk),
    }
    # Use the column order the pipeline was trained with.
    X = pd.DataFrame([row])[list(model.feature_names_in_)]
    price = float(model.predict(X)[0])

    if price <= 0:
        st.warning(
            "These inputs are outside the range the model learned from, so it could not "
            "produce a sensible price. Try a larger area or fewer rooms."
        )
    else:
        st.metric("Estimated price", format_price(price))
        st.caption(f"≈ ₹ {price * 100_000 / sqft:,.0f} per sqft")

    if sqft / bhk < MIN_SQFT_PER_BHK:
        st.warning(
            f"Under {MIN_SQFT_PER_BHK} sqft per bedroom. The model was trained without "
            "homes this cramped, so treat this estimate with caution."
        )
    if bath > bhk + 2:
        st.warning("Many more bathrooms than bedrooms is unusual; the estimate may be unreliable.")
    if location == "other":
        st.info(
            "Areas outside the list share one general category, so the location effect "
            "is averaged and the estimate is less precise."
        )

with st.expander("About this model"):
    st.markdown(
        """
        - A Ridge regression model trained on Bengaluru listings.
        - Inputs: location, total area, bedrooms (BHK) and bathrooms.
        - Prices are in **lakhs** (1 lakh = ₹100,000; 100 lakh = 1 crore).
        - Estimates are indicative only. Asking prices, builder, floor, age and
          amenities are not in the model.
        """
    )