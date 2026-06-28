import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="TPS Material Screener", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

plt.style.use("dark_background")
plt.rcParams.update({
    "axes.facecolor": "#0e1117", 
    "figure.facecolor": "#0e1117",
    "grid.color": "#333333",
    "axes.edgecolor": "#555555"
})

COLORS = {"ablator": "#e74c3c", "ceramic_tile": "#3498db", "blanket": "#2ecc71", "RCC": "#f39c12", "ceramic_ablator": "#9b59b6", "coating": "#1abc9c"}
CLASS_LABELS = {"ablator": "Ablator", "ceramic_tile": "Ceramic Tile", "blanket": "Fibrous Blanket", "RCC": "Carbon Composite", "ceramic_ablator": "Ceramic Ablator", "coating": "Surface Coating"}
MISSION_WEIGHTS = {
    "single_use": {"w_thermal": 0.40, "w_temp": 0.40, "w_cost": 0.15, "w_install": 0.05},
    "multi_use":  {"w_thermal": 0.30, "w_temp": 0.25, "w_cost": 0.20, "w_install": 0.25},
    "budget":     {"w_thermal": 0.20, "w_temp": 0.30, "w_cost": 0.40, "w_install": 0.10},
}
HIGH_K_EXCLUSIONS = {"RCC"}
STAGE1_FEATURES = ["temperature", "temperature_sq", "density", "specific_heat", "vol_heat_cap", "emissivity"]


@st.cache_data(show_spinner=False)
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r'^\\', '', regex=True)
    if "installation_time_hr" in df.columns:
        df = df.rename(columns={"installation_time_hr": "install_time"})
    if "single_use_temp" not in df.columns and "max_use_temp" in df.columns:
        df["single_use_temp"] = df["max_use_temp"]
    if "multi_use_temp" not in df.columns and "max_use_temp" in df.columns:
        df["multi_use_temp"] = df["max_use_temp"]
    df["material"] = df["material"].str.strip()
    anomaly_mask = df["specific_heat"] < 10
    if anomaly_mask.any():
        df.loc[anomaly_mask, "specific_heat"] = np.nan
    return df


@st.cache_data
def build_material_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("material").agg(
        material_class=("material_class", "first"),
        density=("density", "first"),
        specific_heat=("specific_heat", "mean"),
        emissivity=("emissivity", "mean"),
        mean_k=("thermal_conductivity", "mean"),
        cost_per_m2=("cost_per_m2", "first"),
        install_time=("install_time", "first"),
        max_use_temp=("max_use_temp", "first"),
        n_temp_points=("temperature", "count"),
    ).reset_index()


# =============================================================================
# STAGE 1
# =============================================================================
@st.cache_resource
def train_stage1_model(df: pd.DataFrame):
    df_train = df[~df["material"].isin(HIGH_K_EXCLUSIONS) & df["thermal_conductivity"].notna() & df["specific_heat"].notna()].copy()
    df_train["temperature_sq"] = df_train["temperature"] ** 2
    df_train["vol_heat_cap"] = df_train["density"] * df_train["specific_heat"]

    X = df_train[STAGE1_FEATURES].values
    y = np.log1p(df_train["thermal_conductivity"].values)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_sc, y)

    from sklearn.metrics import r2_score
    train_r2 = r2_score(y, model.predict(X_sc))
    return model, scaler, train_r2


def predict_k(model, scaler, temperature, density, specific_heat, emissivity) -> float:
    feats = np.array([[temperature, temperature ** 2, density, specific_heat, density * specific_heat, emissivity]])
    return float(np.expm1(model.predict(scaler.transform(feats))[0]))


def fill_missing_k(df_summary: pd.DataFrame, model, scaler) -> pd.DataFrame:
    df_out = df_summary.copy()
    missing_mask = df_out["mean_k"].isna() & ~df_out["material"].isin(HIGH_K_EXCLUSIONS)
    for idx in df_out[missing_mask].index:
        row = df_out.loc[idx]
        if pd.notna(row["density"]) and pd.notna(row["specific_heat"]) and pd.notna(row["emissivity"]):
            df_out.at[idx, "mean_k"] = predict_k(model, scaler, 900.0, float(row["density"]), float(row["specific_heat"]), float(row["emissivity"]))
            df_out.at[idx, "k_predicted"] = True
            
    if "k_predicted" not in df_out.columns:
        df_out["k_predicted"] = False
    else:
        df_out["k_predicted"] = df_out["k_predicted"].fillna(False)
    return df_out


# =============================================================================
# STAGE 2
# =============================================================================
def normalise_inverse(series: pd.Series) -> pd.Series:
    inv = 1.0 / series.clip(lower=1e-9)
    mn, mx = inv.min(), inv.max()
    return pd.Series(np.ones(len(series)), index=series.index) if mx == mn else (inv - mn) / (mx - mn)


def normalise(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return pd.Series(np.ones(len(series)), index=series.index) if mx == mn else (series - mn) / (mx - mn)


def score_materials(df_summary, mission_temp, mission_type, budget):
    df_s = df_summary[(df_summary["max_use_temp"] >= mission_temp) & (df_summary["cost_per_m2"] <= budget) & df_summary["mean_k"].notna() & df_summary["cost_per_m2"].notna() & df_summary["install_time"].notna()].copy()
    if df_s.empty: return df_s

    df_s["ThermalScore"] = normalise_inverse(df_s["mean_k"])
    df_s["TemperatureScore"] = normalise(((df_s["max_use_temp"] - mission_temp) / max(mission_temp, 1)).clip(lower=0))
    df_s["CostScore"] = normalise_inverse(df_s["cost_per_m2"])
    df_s["InstallationScore"] = normalise_inverse(df_s["install_time"])

    w = MISSION_WEIGHTS[mission_type]
    df_s["Score"] = (w["w_thermal"] * df_s["ThermalScore"] + w["w_temp"] * df_s["TemperatureScore"] + w["w_cost"] * df_s["CostScore"] + w["w_install"] * df_s["InstallationScore"])
    return df_s.sort_values("Score", ascending=False).reset_index(drop=True)

def plot_score_bars(top: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, max(3, len(top) * 0.55)))
    colors = [COLORS.get(c, "#aaa") for c in top["material_class"]]
    ax.barh([top["material"].iloc[i] for i in range(len(top) - 1, -1, -1)], [top["Score"].iloc[i] for i in range(len(top) - 1, -1, -1)], color=[colors[i] for i in range(len(top) - 1, -1, -1)], edgecolor="#222222", linewidth=0.6, alpha=0.9)
    ax.set_xlabel("Composite Mission Score", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_title("Material Ranking", fontweight="bold", fontsize=11)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(handles=[mpatches.Patch(color=COLORS.get(c, "#aaa"), label=CLASS_LABELS.get(c, c)) for c in top["material_class"].unique()], fontsize=8, loc="lower right", frameon=False)
    plt.tight_layout()
    return fig

def plot_score_decomposition(top: pd.DataFrame, mission_type: str) -> plt.Figure:
    w = MISSION_WEIGHTS[mission_type]
    components = {"Thermal Insulation": w["w_thermal"] * top["ThermalScore"].values, "Temp. Margin": w["w_temp"] * top["TemperatureScore"].values, "Cost Efficiency": w["w_cost"] * top["CostScore"].values, "Installation": w["w_install"] * top["InstallationScore"].values}
    comp_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    fig, ax = plt.subplots(figsize=(max(8, len(top) * 1.1), 4))
    bottom = np.zeros(len(top))
    for (label, vals), col in zip(components.items(), comp_colors):
        ax.bar(top["material"], vals, bottom=bottom, label=label, color=col, alpha=0.85, edgecolor="#222222", linewidth=0.5)
        bottom += vals
    ax.set_ylabel("Weighted Score Contribution", fontsize=10, fontweight="bold")
    ax.set_title("Score Decomposition — What Drives the Ranking?", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=22, ha="right", fontsize=9)
    plt.tight_layout()
    return fig

def plot_k_vs_temp(df_raw: pd.DataFrame, selected_material: str) -> plt.Figure:
    subset = df_raw[df_raw["material"] == selected_material].sort_values("temperature")
    mat_class = subset["material_class"].iloc[0] if not subset.empty else "unknown"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(subset["temperature"], subset["thermal_conductivity"], marker="o", color=COLORS.get(mat_class, "#bbbbbb"), linewidth=2, markersize=6)
    ax.set_xlabel("Temperature (K)", fontweight="bold")
    ax.set_ylabel("Thermal Conductivity (W/m·K)", fontweight="bold")
    ax.set_title(f"k–T Curve: {selected_material}", fontweight="bold")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    return fig

@st.cache_resource
def load_stage1_model():

    import joblib
    import os

    model = joblib.load(
        os.path.join(
            os.path.dirname(__file__),
            "models",
            "ridge_stage1.pkl"
        )
    )
    scaler = joblib.load(
        os.path.join(
            os.path.dirname(__file__),
            "models",
            "scaler_stage1.pkl"
        )
    )
    return model, scaler
# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nasa_tps_materials.csv")
    if not os.path.exists(csv_path):
        st.error("Dataset not found. Place **nasa_tps_materials.csv** in the same directory.")
        st.stop()

    df_raw = load_and_preprocess(csv_path)
    df_summary = build_material_summary(df_raw)

    with st.spinner("Loading ML pipeline..."):
        model_s1, scaler_s1 = load_stage1_model()
       
    df_summary = fill_missing_k(df_summary, model_s1, scaler_s1)
    n_predicted = df_summary["k_predicted"].sum() if "k_predicted" in df_summary.columns else 0

    st.title("🚀 NASA TPS Material Screening Tool")
    st.markdown("**Two-Stage Machine Learning Pipeline** for Thermal Protection System Selection")
    st.markdown("---")

    st.sidebar.header("⚙️ Mission Constraints")
    mission_temp = st.sidebar.slider("Max Mission Temp (K)", 300, 2000, 1500, 50)
    budget = st.sidebar.slider("Budget per m² (USD)", 500, 200_000, 15_000, 500)
    mission_type = st.sidebar.selectbox("Mission Type", ["single_use", "multi_use", "budget"], format_func=lambda x: {"single_use": "Single-Use Ablative", "multi_use": "Reusable (Multi-Use)", "budget": "Budget Constrained"}[x])
    top_n = st.sidebar.slider("Show Top N Materials", 3, 12, 6)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Active Weight Profile**")
    w = MISSION_WEIGHTS[mission_type]
    st.sidebar.progress(w["w_thermal"], text=f"Thermal: {w['w_thermal']:.0%}")
    st.sidebar.progress(w["w_temp"], text=f"Temp Margin: {w['w_temp']:.0%}")
    st.sidebar.progress(w["w_cost"], text=f"Cost: {w['w_cost']:.0%}")
    st.sidebar.progress(w["w_install"], text=f"Install: {w['w_install']:.0%}")

    with st.expander("ℹ️ Methodology & Scope", expanded=False):
        st.markdown(f"""
        **Stage 1 — Thermal Conductivity Prediction** Model: Ridge Regression | Features: temperature, temperature², density, specific heat, volumetric heat capacity (ρ·Cₚ), emissivity  
        Stage 1 Model: **Pre-trained Ridge Regression**** Missing values filled: **{n_predicted}** material entries  

        **Stage 1 Scope Restriction** RCC and carbon composites are excluded from prediction because their thermal conductivity (9.5–12.5 W/m·K) is ~52× higher than insulating TPS materials (0.03–0.50 W/m·K). A single model cannot bridge this gap reliably. RCC materials are retained in Stage 2 with their measured properties.

        **Stage 2 — Mission Scoring** Hard filter: temperature limit ≥ mission temp AND cost ≤ budget  
        Soft score: weighted sum of thermal insulation, temperature margin, cost efficiency, and installation efficiency  
        Weights are mission-type dependent (single-use, multi-use, budget-constrained).
        """)

    st.subheader("Mission-Aware Material Ranking")
    ranked = score_materials(df_summary, mission_temp, mission_type, budget)
    if ranked.empty:
        st.warning("No materials meet constraints. Adjust temperature or budget in the sidebar.")
    else:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.dataframe(ranked[["material", "material_class", "mean_k", "max_use_temp", "cost_per_m2", "Score"]].head(top_n).style.format({"mean_k": "{:.4f} W/m·K", "max_use_temp": "{:,.0f} K", "cost_per_m2": "${:,.0f}", "Score": "{:.4f}"}).background_gradient(subset=["Score"], cmap="viridis", vmin=0, vmax=1), use_container_width=True, hide_index=True)
        with col2:
            st.pyplot(plot_score_bars(ranked.head(top_n)))
            
        st.subheader("🔍 Score Decomposition")
        st.pyplot(plot_score_decomposition(ranked.head(top_n), mission_type))
        
        st.subheader("📈 Thermal Conductivity vs Temperature")
        available = [m for m in ranked.head(top_n)["material"].tolist() if m in df_raw["material"].values]
        if available:
            selected = st.selectbox("Select material to inspect:", available)
            st.pyplot(plot_k_vs_temp(df_raw, selected))
    
        st.markdown("---")
        st.subheader("📥 Export Results")
        txt_report = "Top 5 TPS Material Recommendations\n" + "="*40 + "\n\n"
        for idx, row in ranked.head(5).iterrows():
            txt_report += f"Rank {idx+1}: {row['material']} ({CLASS_LABELS.get(row['material_class'], row['material_class'])})\n  • Composite Score : {row['Score']:.4f}\n  • Thermal Cond.   : {row['mean_k']:.4f} W/m·K\n  • Max Temp Limit  : {row['max_use_temp']:.0f} K\n  • Cost Estimate   : ${row['cost_per_m2']:,.0f} / m²\n\n"
        st.download_button(
            label="📄 Download Top 5 Recommendations (TXT)", 
            data=txt_report, 
            file_name="top_5_tps_recommendations.txt", 
            mime="text/plain"
        )
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888888;'><p style='font-size: 1.1em; font-weight: 600; margin-bottom: 0px;'>Developed by Akanshha Sonowal</p><p style='font-size: 0.9em; margin-top: 5px;'>Machine Learning • Materials Informatics • NASA TPS Screening</p></div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()