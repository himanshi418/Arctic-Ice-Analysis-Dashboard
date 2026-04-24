
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Arctic Ice Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL CSS  – polar-noir theme
# ─────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root palette ── */
    :root {
        --bg:        #050d1a;
        --surface:   #0a1628;
        --card:      #0d1e35;
        --border:    rgba(0,200,255,0.12);
        --accent:    #00c8ff;
        --accent2:   #00ffc8;
        --accent3:   #7b8cff;
        --text:      #cde6f5;
        --muted:     #5a7a99;
        --glow:      0 0 24px rgba(0,200,255,0.25);
    }

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ── App wrapper ── */
    .stApp {
        background: radial-gradient(ellipse at 20% 0%, #071830 0%, var(--bg) 60%) !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Syne', sans-serif !important;
    }

    /* Sidebar nav highlight */
    div[data-testid="stRadio"] label:has(input:checked) {
        background: rgba(0,200,255,0.1) !important;
        border-left: 3px solid var(--accent) !important;
        border-radius: 4px !important;
        padding-left: 8px !important;
        color: var(--accent) !important;
    }

    /* ── Hero header ── */
    .hero-wrap {
        text-align: center;
        padding: 2.6rem 0 1.6rem;
        margin-bottom: 0.5rem;
    }
    .hero-wrap h1 {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00c8ff 0%, #00ffc8 55%, #7b8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }
    .hero-wrap p {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* ── Section headings ── */
    h2, h3, .stSubheader {
        font-family: 'Syne', sans-serif !important;
        color: var(--accent) !important;
        letter-spacing: -0.5px;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--accent);
        border-left: 3px solid var(--accent);
        padding-left: 10px;
        margin: 1.8rem 0 0.8rem;
        letter-spacing: 0.3px;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.2rem !important;
        box-shadow: var(--glow) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 36px rgba(0,200,255,0.35) !important;
    }
    div[data-testid="metric-container"] label {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        color: var(--muted) !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
        color: var(--accent) !important;
        font-weight: 700 !important;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    .stDataFrame thead tr th {
        background: var(--surface) !important;
        color: var(--accent) !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stDataFrame tbody tr {
        background: var(--card) !important;
    }
    .stDataFrame tbody tr:hover td {
        background: rgba(0,200,255,0.06) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #050d1a !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.6rem !important;
        transition: opacity 0.2s, transform 0.15s;
        box-shadow: 0 0 18px rgba(0,200,255,0.3) !important;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: scale(1.03);
    }

    /* ── Success / Warning / Error boxes ── */
    .stSuccess { background: rgba(0,255,200,0.08) !important; border-left-color: var(--accent2) !important; border-radius: 10px !important; }
    .stWarning { background: rgba(255,180,0,0.08) !important; border-radius: 10px !important; }
    .stError   { background: rgba(255,80,80,0.08) !important; border-radius: 10px !important; }

    /* ── Sliders ── */
    .stSlider > div > div > div > div {
        background: var(--accent) !important;
    }

    /* ── Number input ── */
    .stNumberInput input {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }

    /* ── Multiselect ── */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: rgba(0,200,255,0.15) !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
        border-radius: 6px !important;
    }
    .stMultiSelect [data-baseweb="select"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }

    /* ── Plotly chart container ── */
    .js-plotly-plot { border-radius: 14px; overflow: hidden; }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
        font-family: 'Syne', sans-serif !important;
        border-radius: 8px !important;
        transition: background 0.2s;
    }
    .stDownloadButton > button:hover {
        background: rgba(0,200,255,0.1) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* ── Info badge ── */
    .badge {
        display: inline-block;
        background: rgba(0,200,255,0.1);
        border: 1px solid var(--border);
        color: var(--accent);
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        border-radius: 20px;
        padding: 3px 12px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
#  PLOTLY TEMPLATE
# ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,22,40,0.0)",
    plot_bgcolor="rgba(10,22,40,0.0)",
    font=dict(family="DM Sans, sans-serif", color="#cde6f5"),
    title_font=dict(family="Syne, sans-serif", size=17, color="#00c8ff"),
    xaxis=dict(gridcolor="rgba(0,200,255,0.07)", zerolinecolor="rgba(0,200,255,0.1)"),
    yaxis=dict(gridcolor="rgba(0,200,255,0.07)", zerolinecolor="rgba(0,200,255,0.1)"),
    margin=dict(t=50, b=40, l=40, r=20),
    hoverlabel=dict(bgcolor="#0d1e35", bordercolor="#00c8ff", font_color="#cde6f5"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,200,255,0.15)"),
)

def apply_layout(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title_text=title)
    return fig


# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("Arctic_Ice_Data.csv")
df.columns = df.columns.str.strip()

# ─────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────
df["Date"] = pd.to_datetime(
    df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-" + df["Day"].astype(str),
    errors="coerce",
)
df["Day_of_Year"] = df["Date"].dt.dayofyear
df["Season"] = df["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring",  4: "Spring",  5: "Spring",
    6: "Summer",  7: "Summer",  8: "Summer",
    9: "Autumn",  10: "Autumn", 11: "Autumn",
})
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
df["Month_Name"] = df["Month"].map(month_names)

# ─────────────────────────────────────────
#  AUTO-DETECT COLUMNS
# ─────────────────────────────────────────
extent_col = "Extent"
concentration_col = None
temperature_col = None

for col in df.columns:
    low = col.lower()
    if "extent"        in low: extent_col        = col
    if "concentration" in low: concentration_col = col
    if "temperature"   in low or "temp" in low: temperature_col = col

# ─────────────────────────────────────────
#  ML MODEL
# ─────────────────────────────────────────
features = ["Year", "Month", "Day_of_Year"]
ml_df = df.dropna(subset=features + [extent_col])
X, y = ml_df[features], ml_df[extent_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <span style='font-family:Syne,sans-serif; font-size:1.5rem; font-weight:800;
                     background:linear-gradient(135deg,#00c8ff,#00ffc8);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        ❄️ Arctic Ice
        </span>
        <p style='color:#5a7a99; font-size:0.72rem; letter-spacing:2px;
                  text-transform:uppercase; margin:2px 0 0;'>Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Trend Analysis", "Distribution Charts",
     "Season Analysis", "Climate Analysis", "ML Prediction", "Dataset"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-family:Syne,sans-serif; font-size:0.72rem; text-transform:uppercase;"
    "letter-spacing:1.5px; color:#5a7a99;'>Filters</p>",
    unsafe_allow_html=True,
)

years   = sorted(df["Year"].dropna().unique())
months  = sorted(df["Month"].dropna().unique())
seasons = sorted(df["Season"].dropna().unique())

selected_years   = st.sidebar.multiselect("Years",   years,   default=years)
selected_months  = st.sidebar.multiselect("Months",  months,  default=months)
selected_seasons = st.sidebar.multiselect("Seasons", seasons, default=seasons)

filtered_df = df[
    df["Year"].isin(selected_years) &
    df["Month"].isin(selected_months) &
    df["Season"].isin(selected_seasons)
]

# ─────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────
st.markdown(
    """
    <div class="hero-wrap">
        <div class="badge">Arctic Climate Intelligence</div>
        <h1>Arctic Ice Dashboard</h1>
        <p>Interactive polar ice analysis · machine-learning predictions · climate trends</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════

# ── OVERVIEW ──────────────────────────────
if page == "Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(filtered_df):,}")
    col2.metric("Total Columns",   filtered_df.shape[1])
    col3.metric("Start Year",      int(filtered_df["Year"].min()))
    col4.metric("End Year",        int(filtered_df["Year"].max()))

    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(), use_container_width=True)

# ── TREND ANALYSIS ────────────────────────
elif page == "Trend Analysis":
    st.markdown('<div class="section-title">Average Ice Extent by Year</div>', unsafe_allow_html=True)
    yearly_extent = filtered_df.groupby("Year")[extent_col].mean().reset_index()

    fig1 = px.line(yearly_extent, x="Year", y=extent_col, markers=True)
    fig1.update_traces(
        line=dict(width=3, color="#00c8ff"),
        marker=dict(size=8, color="#00ffc8", line=dict(color="#050d1a", width=2)),
        fill="tozeroy",
        fillcolor="rgba(0,200,255,0.05)",
    )
    apply_layout(fig1, "Average Ice Extent by Year")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="section-title">Average Ice Extent by Month</div>', unsafe_allow_html=True)
    monthly_extent = filtered_df.groupby(["Month", "Month_Name"])[extent_col].mean().reset_index()
    monthly_extent = monthly_extent.sort_values("Month")

    fig2 = px.bar(monthly_extent, x="Month_Name", y=extent_col,
                  color=extent_col, color_continuous_scale=["#0a1628","#00c8ff","#00ffc8"])
    apply_layout(fig2, "Average Ice Extent by Month")
    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(fig2, use_container_width=True)

# ── DISTRIBUTION ──────────────────────────
elif page == "Distribution Charts":
    st.markdown('<div class="section-title">Distribution of Ice Extent</div>', unsafe_allow_html=True)
    fig3 = px.histogram(filtered_df, x=extent_col, nbins=40,
                        color_discrete_sequence=["#00c8ff"])
    apply_layout(fig3, "Distribution of Ice Extent")
    fig3.update_traces(marker_line_color="#050d1a", marker_line_width=0.5, opacity=0.85)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-title">Boxplot</div>', unsafe_allow_html=True)
    fig4 = px.box(filtered_df, y=extent_col, color_discrete_sequence=["#7b8cff"])
    apply_layout(fig4, "Ice Extent Boxplot")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_df = filtered_df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig5 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     width=None, height=600)
    apply_layout(fig5, "Correlation Heatmap")
    fig5.update_layout(margin=dict(t=60, b=60, l=80, r=80))
    st.plotly_chart(fig5, use_container_width=True)

# ── SEASON ────────────────────────────────
elif page == "Season Analysis":
    season_colors = ["#00c8ff","#00ffc8","#7b8cff","#ffb347"]
    season_count  = filtered_df["Season"].value_counts().reset_index()
    season_count.columns = ["Season", "Count"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Seasonal Distribution</div>', unsafe_allow_html=True)
        fig6 = px.pie(season_count, names="Season", values="Count", hole=0.58,
                      color_discrete_sequence=season_colors)
        apply_layout(fig6, "Seasonal Distribution")
        fig6.update_traces(textfont_color="#cde6f5",
                           marker=dict(line=dict(color="#050d1a", width=2)))
        st.plotly_chart(fig6, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Records by Season</div>', unsafe_allow_html=True)
        fig8 = px.bar(season_count, x="Season", y="Count", color="Season",
                      color_discrete_sequence=season_colors)
        apply_layout(fig8, "Count of Records by Season")
        fig8.update_traces(marker_line_width=0)
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="section-title">Ice Extent Distribution by Season</div>', unsafe_allow_html=True)
    fig9 = px.violin(filtered_df, x="Season", y=extent_col, color="Season",
                     box=True, points=False, color_discrete_sequence=season_colors)
    apply_layout(fig9, "Ice Extent Distribution by Season")
    st.plotly_chart(fig9, use_container_width=True)

# ── CLIMATE ───────────────────────────────
elif page == "Climate Analysis":
    if concentration_col is not None:
        st.markdown('<div class="section-title">Ice Concentration & Extent by Year</div>', unsafe_allow_html=True)
        grouped = filtered_df.groupby("Year")[[concentration_col, extent_col]].mean().reset_index()
        fig7 = px.bar(grouped, x="Year", y=[concentration_col, extent_col],
                      barmode="group", color_discrete_sequence=["#00c8ff","#00ffc8"])
        apply_layout(fig7, "Ice Concentration & Extent by Year")
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.warning("Ice Concentration column not found in dataset.")

    if temperature_col is not None:
        st.markdown('<div class="section-title">Average Sea Temperature by Month</div>', unsafe_allow_html=True)
        temp_month = filtered_df.groupby(["Month","Month_Name"])[temperature_col].mean().reset_index()
        temp_month = temp_month.sort_values("Month")
        fig10 = px.line(temp_month, x="Month_Name", y=temperature_col, markers=True)
        fig10.update_traces(
            line=dict(width=3, color="#ff5e5e"),
            marker=dict(size=9, color="#ffb347", line=dict(color="#050d1a", width=2)),
        )
        apply_layout(fig10, "Average Sea Temperature by Month")
        st.plotly_chart(fig10, use_container_width=True)
    else:
        st.warning("Sea Temperature column not found in dataset.")

# ── ML PREDICTION ─────────────────────────
elif page == "ML Prediction":
    st.markdown('<div class="section-title">Random Forest Model Performance</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("R² Score",               round(r2, 3))
    c2.metric("Mean Absolute Error",    round(mae, 3))

    st.markdown('<div class="section-title">Predict Ice Extent</div>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    year_input  = pc1.number_input("Year",  min_value=1970, max_value=2050, value=2025)
    month_input = pc2.slider("Month", min_value=1, max_value=12, value=6)
    day_input   = pc3.slider("Day",   min_value=1, max_value=31, value=15)

    input_date = pd.to_datetime(
        f"{int(year_input)}-{int(month_input)}-{int(day_input)}", errors="coerce"
    )

    if pd.isna(input_date):
        st.error("Invalid date combination.")
    else:
        if st.button("✦ Predict Ice Extent"):
            prediction = model.predict([[year_input, month_input, input_date.dayofyear]])
            st.success(f"Predicted Arctic Ice Extent: **{round(prediction[0], 3)}** million km²")

    st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
    pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    fig_ml = px.scatter(pred_df, x="Actual", y="Predicted",
                        color_discrete_sequence=["#00c8ff"], opacity=0.65)
    # perfect-fit reference line
    mn, mx = pred_df["Actual"].min(), pred_df["Actual"].max()
    fig_ml.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                line=dict(color="#00ffc8", width=2, dash="dash"),
                                name="Perfect fit"))
    apply_layout(fig_ml, "Actual vs Predicted Ice Extent")
    st.plotly_chart(fig_ml, use_container_width=True)

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    importance_df = pd.DataFrame({
        "Feature":    features,
        "Importance": model.feature_importances_,
    })
    fig_imp = px.bar(importance_df, x="Feature", y="Importance",
                     color="Importance", color_continuous_scale=["#0d1e35","#00c8ff","#00ffc8"])
    fig_imp.update_traces(marker_line_width=0)
    apply_layout(fig_imp, "Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

# ── DATASET ───────────────────────────────
elif page == "Dataset":
    st.markdown('<div class="section-title">Full Processed Dataset</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Processed CSV",
        data=csv,
        file_name="processed_arctic_dataset.csv",
        mime="text/csv",
    )
    
