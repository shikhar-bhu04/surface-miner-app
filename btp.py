import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Config & small CSS for card + centering
# -------------------------------
st.set_page_config(page_title="Surface Miner Applicability (CI-based)", layout="wide")

# Small CSS for card-like UI and centered content
CARD_CSS = """
<style>
.result-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.06);
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.4);
  max-width: 720px;
  margin: 0 auto;
}
.badge {
  display:inline-block;
  padding:10px 18px;
  border-radius:999px;
  font-weight:700;
  font-size:20px;
  color: #111827;
  margin-bottom:8px;
}
.ci-metric {
  font-size:18px;
  color: #e6e6e6;
  margin-top:6px;
}
.center {
  text-align:center;
}
.small-muted {
  color: rgba(230,230,230,0.7);
  font-size:13px;
}
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

# -------------------------------
# Machine model -> power lookup (from your machine table)
# -------------------------------
machine_models = {
    # Wirtgen GmbH
    "SM2100": 448, "Wirtgen SM2100": 448,
    "SM2200": 671, "Wirtgen SM2200": 671,
    "SM2500": 783, "Wirtgen SM2500": 783,
    "SM3500": 895, "Wirtgen SM3500": 895,
    "SM4200": 1194, "Wirtgen SM4200": 1194,
    # Vermeer
    "T855": 281, "Vermeer T855": 281,
    "T955": 309, "Vermeer T955": 309,
    "T1055": 317, "Vermeer T1055": 317,
    "T1255": 447, "Vermeer T1255": 447,
    # L & T
    "KSM223": 597, "L&T KSM223": 597,
    "KSM304": 895, "L&T KSM304": 895,
    # TAKRAF GmbH
    "MTS180": 500, "TAKRAF MTS180": 500,
    "MTS300": 750, "TAKRAF MTS300": 750,
    "MTS500": 1650, "TAKRAF MTS500": 1650,
    "MTS800": 2000, "TAKRAF MTS800": 2000,
    "MTS1250": 2500, "TAKRAF MTS1250": 2500,
    "MTS2000": 2500, "TAKRAF MTS2000": 2500,
    # Bitelli
    "SF202": 515, "Bitelli SF202": 515
}

# -------------------------------
# Rating mapping functions
# -------------------------------
def rating_PLI(pli):
    if pli < 0.5:
        return 5
    elif pli < 1.5:
        return 10
    elif pli < 2.0:
        return 15
    elif pli < 3.5:
        return 20
    else:
        return 25

def rating_Jv(jv):
    if pd.isna(jv):
        return 15
    try:
        jv = float(jv)
    except:
        return 15
    if jv > 30:
        return 5
    elif jv > 10:
        return 10
    elif jv > 3:
        return 15
    elif jv > 1:
        return 20
    else:
        return 25

def rating_abrasivity(aw):
    if pd.isna(aw):
        return 9
    try:
        aw = float(aw)
    except:
        return 9
    if aw < 0.5:
        return 3
    elif aw < 1.0:
        return 6
    elif aw < 2.0:
        return 9
    elif aw < 3.0:
        return 12
    else:
        return 15

def rating_direction(direction_input):
    if pd.isna(direction_input):
        return 9
    val = str(direction_input).strip().lower()
    # common textual inputs
    if val in ("parallel", "para", "0", "0°", "0deg"):
        return 15
    if val in ("perpendicular", "perp", "90", "90°", "90deg"):
        return 3
    # numeric angle
    try:
        angle = float(val)
        angle = abs(angle) % 180
        if 72 <= angle <= 90:
            return 3
        elif 54 <= angle < 72:
            return 6
        elif 36 <= angle < 54:
            return 9
        elif 18 <= angle < 36:
            return 12
        else:
            return 15
    except:
        return 9

def rating_machine_power(power_kw):
    if pd.isna(power_kw):
        return 12
    try:
        p = float(power_kw)
    except:
        return 12
    if p > 1000:
        return 4
    elif p >= 800:
        return 8
    elif p >= 600:
        return 12
    elif p >= 400:
        return 16
    else:
        return 20

def derive_power_from_model(model_name):
    if pd.isna(model_name):
        return np.nan
    key = str(model_name).strip()
    if key in machine_models:
        return machine_models[key]
    for k in machine_models:
        if key.lower() in k.lower() or k.lower() in key.lower():
            return machine_models[k]
    return np.nan

def classify_CI(ci):
    if ci < 50:
        return "Very Suitable"
    elif ci < 60:
        return "Suitable"
    elif ci < 70:
        return "Borderline"
    elif ci < 80:
        return "Not Suitable"
    else:
        return "Should NOT be Deployed"

def badge_props(suitability_label):
    # returns (emoji, background_color, text_color)
    if suitability_label == "Very Suitable":
        return "🟢", "#6EE7B7", "#064E3B"
    elif suitability_label == "Suitable":
        return "🟡", "#FDE68A", "#92400E"
    elif suitability_label == "Borderline":
        return "🟠", "#FDBA74", "#7C2D12"
    elif suitability_label == "Not Suitable":
        return "🔴", "#FCA5A5", "#7F1D1D"
    else:
        return "🔴", "#FCA5A5", "#7F1D1D"

# -------------------------------
# App header & description
# -------------------------------
st.title("🚜 Surface Miner Applicability — CI-based Calculator")
st.markdown("Compute **Cuttability Index (CI)** from raw inputs (Is + Jv + Aw + Js + M) and determine surface miner suitability.")
st.write("")  # spacer

tab1, tab2 = st.tabs(["🔧 Manual Input", "📁 Predict from CSV"])

# -------------------------------
# MANUAL INPUT TAB
# -------------------------------
with tab1:
    # Sidebar inputs
    st.sidebar.header("Input rock & machine parameters (raw)")
    pli = st.sidebar.number_input("Point Load Index (PLI), MPa", min_value=0.0, value=1.5, step=0.1, format="%.2f")
    jv = st.sidebar.number_input("Volumetric Joint Count (Jv), no/m³", min_value=0.0, value=25.0, step=1.0, format="%.2f")
    aw = st.sidebar.number_input("Abrasivity (Aw)", min_value=0.0, value=0.6, step=0.1, format="%.2f")

    st.sidebar.markdown("**Direction of cutting vs major joint set**")
    st.sidebar.write("Choose Parallel/Perpendicular or enter angle (°) between cutting direction and main joint set.")
    direction_option = st.sidebar.selectbox("Direction input type", ["Parallel / Perpendicular", "Angle (deg)"])
    if direction_option == "Parallel / Perpendicular":
        direction = st.sidebar.selectbox("Direction", ["Parallel", "Perpendicular"])
        direction_val = direction
    else:
        direction_angle = st.sidebar.number_input("Angle (degrees, 0–90)", min_value=0.0, max_value=90.0, value=90.0, step=1.0)
        direction_val = float(direction_angle)

    st.sidebar.markdown("### Machine selection")
    model_list = ["-- Select model (optional) --"] + sorted(list(set(machine_models.keys())))
    selected_model = st.sidebar.selectbox("Machine model (optional)", model_list, index=0)
    if selected_model != "-- Select model (optional) --":
        derived_power = derive_power_from_model(selected_model)
        # if derive_power returns nan allow manual override (displayed)
        if np.isnan(derived_power):
            derived_power = st.sidebar.number_input("Machine power (kW) — model not matched, enter value", min_value=0.0, value=800.0, step=1.0)
    else:
        derived_power = st.sidebar.number_input("Machine power (kW) — optional (used if model not chosen)", min_value=0.0, value=800.0, step=1.0)

    # compute ratings
    Is = rating_PLI(pli)
    Jv_rating = rating_Jv(jv)
    Aw_rating = rating_abrasivity(aw)
    Js = rating_direction(direction_val)
    M = rating_machine_power(derived_power)
    CI = Is + Jv_rating + Aw_rating + Js + M
    suitability = classify_CI(CI)
    emoji, bg_color, txt_color = badge_props(suitability)

    # ------------------------------
    # TOP: Centered card with badge + CI metric
    # ------------------------------
    badge_html = f"""
    <div class="result-card center">
      <div style="display:flex; align-items:center; justify-content:center; flex-direction:column;">
        <div class="badge" style="background:{bg_color}; color:{txt_color};">
          {emoji} &nbsp; {suitability}
        </div>
        <div class="ci-metric">Cuttability Index (CI): <strong style="font-size:22px; color:#ffffff;">{int(CI)}</strong></div>
        
      </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

    st.write("")  # small spacer

    # ------------------------------
    # BELOW: Detailed ratings and chart (two-column)
    # ------------------------------
    st.subheader("Rating Breakdown")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric("Point Load Index (PLI), MPa", f"{pli:.2f}")
        st.metric("PLI rating (Is)", Is)
        st.metric("Volumetric Joint Count (Jv), no/m³", f"{jv:.2f}")
        st.metric("Jv rating", Jv_rating)

    with col2:
        st.metric("Abrasivity (Aw)", f"{aw:.2f}")
        st.metric("Aw rating", Aw_rating)
        st.metric("Direction (input)", str(direction_val))
        st.metric("Direction rating (Js)", Js)

    st.metric("Derived Machine Power (kW)", f"{derived_power:.0f}")
    st.metric("Machine Power rating (M)", M)

    # compact bar chart of components
    components = {
        "Is (PLI)": Is,
        "Jv": Jv_rating,
        "Aw": Aw_rating,
        "Js (dir)": Js,
        "M (power)": M
    }
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.barh(list(components.keys()), list(components.values()))
    ax.set_xlabel("Rating value", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

    st.markdown("---")
    st.markdown("**Notes:** CI = Is + Jv + Aw + Js + M. The table above shows individual component ratings used to compute CI.")

# -------------------------------
# CSV BATCH PREDICTION TAB
# -------------------------------
with tab2:
    st.subheader("📂 Batch prediction from CSV (raw params)")
    st.markdown("CSV should contain columns (any subset): `PointLoadIndex` (PLI MPa), `VolumetricJointCount`, `Abrasivity`, `DirectionOfCutting` (or `DirectionAngle`), `MachineModel` or `MachinePower`.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview (first rows)")
        st.dataframe(df.head())

        # derive machine power
        model_cols = ["MachineModel", "Machine", "Model"]
        found_model_col = None
        for c in model_cols:
            if c in df.columns:
                found_model_col = c
                break

        if found_model_col:
            df["DerivedMachinePower"] = df[found_model_col].apply(derive_power_from_model)
        else:
            df["DerivedMachinePower"] = np.nan

        if "MachinePower" in df.columns:
            df["DerivedMachinePower"] = df["DerivedMachinePower"].fillna(df["MachinePower"])

        df["DerivedMachinePower"] = df["DerivedMachinePower"].fillna(800)

        # direction parsing
        if "DirectionAngle" in df.columns:
            df["DirectionParsed"] = df["DirectionAngle"]
        elif "DirectionOfCutting" in df.columns:
            df["DirectionParsed"] = df["DirectionOfCutting"]
        else:
            df["DirectionParsed"] = "Perpendicular"

        # compute per-row ratings & CI
        def compute_row_ratings(r):
            pli_r = r.get("PointLoadIndex", np.nan)
            jv_r = r.get("VolumetricJointCount", np.nan)
            aw_r = r.get("Abrasivity", np.nan)
            dir_r = r.get("DirectionParsed", np.nan)
            mp_r = r.get("DerivedMachinePower", np.nan)

            Is_r = rating_PLI(pli_r if not pd.isna(pli_r) else 1.5)
            Jv_r = rating_Jv(jv_r if not pd.isna(jv_r) else 25)
            Aw_r = rating_abrasivity(aw_r if not pd.isna(aw_r) else 0.6)
            Js_r = rating_direction(dir_r)
            M_r = rating_machine_power(mp_r if not pd.isna(mp_r) else 800)

            CI_r = Is_r + Jv_r + Aw_r + Js_r + M_r
            Suit_r = classify_CI(CI_r)
            return pd.Series({
                "Is": Is_r,
                "Jv_rating": Jv_r,
                "Aw_rating": Aw_r,
                "Js": Js_r,
                "M_rating": M_r,
                "DerivedMachinePower": mp_r,
                "CI": CI_r,
                "Suitability": Suit_r
            })

        ratings_df = df.apply(compute_row_ratings, axis=1)
        out = pd.concat([df.reset_index(drop=True), ratings_df.reset_index(drop=True)], axis=1)

        # show small top summary: distribution counts + color legend
        st.markdown("### Suitability distribution")
        counts = out["Suitability"].value_counts().reindex(["Very Suitable","Suitable","Borderline","Not Suitable","Should NOT be Deployed"]).fillna(0)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        counts.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Count", fontsize=8)
        ax2.set_xlabel("Suitability", fontsize=8)
        ax2.tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=False)

        # show results table (with CI & Suitability)
        display_cols = [c for c in ["Mine","PointLoadIndex","VolumetricJointCount","Abrasivity","DirectionOfCutting","DirectionAngle","DerivedMachinePower","Is","Jv_rating","Aw_rating","Js","M_rating","CI","Suitability"] if c in out.columns]
        st.markdown("### Results (first 200 rows)")
        st.dataframe(out[display_cols].head(200))

        # download
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results (CSV)", csv, "ci_results.csv", "text/csv")
    else:
        st.info("Upload a CSV file containing raw parameters as described above.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("**Developed by Shikhar Prajapati — IIT (BHU)**  |  Mining Engineering Dept.")
