import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import utils  # Our shared logic
import json
import os

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="DelAgua Stove Adoption Analytics",
    page_icon="🔥",
    layout="wide"
)

# Updated Styling for visibility in Dark Mode
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #464855;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('models/stove_adoption_model.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    with open('models/model_metadata.json', 'r') as f:
        meta = json.load(f)
    # Load sample data for visualizations
    df = pd.read_csv('data/processed_stove_data.csv')
    return model, encoder, meta, df

try:
    model, encoder, meta, sample_df = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run 'python train_model.py' first.")
    st.stop()

# 3. SIDEBAR NAVIGATION
st.sidebar.title("🛠️ Program Tools")
page = st.sidebar.radio("Navigation", 
    ["Executive Dashboard", "Risk Predictor (Single)", "Batch Intelligence", "Model Performance"])

# --- PAGE: EXECUTIVE DASHBOARD (Updated with Stratified Filter) ---
if page == "Executive Dashboard":
    st.title("📊 Program Executive Dashboard")
    
    # 1. NEW FEATURE: Balanced Data Selector
    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Data View Settings")
    data_pct = st.sidebar.slider("Analysis Coverage (%)", 10, 100, 100)
    
    # Logic for Balanced/Stratified Sampling
    # This ensures we take an equal percentage from every district
    balanced_df = sample_df.groupby('district', group_keys=False).apply(
        lambda x: x.sample(frac=data_pct/100, random_state=42)
    )

    # KPI Row (Using balanced_df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model ROC-AUC", f"{meta['roc_auc']:.3f}")
    with col2:
        st.metric("Avg. Adoption Risk", f"{balanced_df['low_adoption'].mean()*100:.1f}%")
    with col3:
        risk_map = balanced_df.groupby('district')['low_adoption'].mean()
        st.metric("Highest Risk Region", risk_map.idxmax())
    with col4:
        st.metric("Active Records", len(balanced_df))

    st.divider()

    # Visualizations Row
    c1, c2 = st.columns([1.2, 1]) # Map slightly wider
    
    with c1:
        st.subheader("📍 Regional Adoption Map")
        # Theme changed to 'open-street-map' for more color/detail
        fig = px.scatter_mapbox(balanced_df, lat="latitude", lon="longitude", 
                                color="low_adoption", size="market_climb_index",
                                color_continuous_scale="RdYlGn_r", zoom=8.5, height=550,
                                labels={'low_adoption': 'Risk Level'},
                                hover_name="district")
        
        # 'open-street-map' provides the colourful Google-style detail
        fig.update_layout(mapbox_style="open-street-map") 
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("⛰️ Boxplot Interpretation")
        # The Boxplot
        fig_box = px.box(balanced_df, x="low_adoption", y="market_climb_index", 
                         color="low_adoption", points="all",
                         color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                         labels={"low_adoption": "Adoption Status"})
        fig_box.update_xaxes(tickvals=[0, 1], ticktext=["Success", "At Risk"])
        st.plotly_chart(fig_box, use_container_width=True)
        
        # 2. BOXPLOT INTERPRETATION FOR MANAGERS
        st.info("""
        **How to read this chart:**
        * **The Vertical Position:** Notice the Red Box (At Risk) is higher than the Green Box. This means households that fail to adopt have a much higher 'Market Climb Index'.
        * **The Conclusion:** Physical isolation (distance + high altitude) is a 'hard barrier.' If the Index is above 15, households are significantly more likely to quit using the stove because the effort to get fuel is too high.
        """)
        

# --- PAGE 2: RISK PREDICTOR (Single) ---
elif page == "Risk Predictor (Single)":
    st.title("🔮 Household Risk Oracle")
    st.markdown("Select a district to auto-fill geographic coordinates.")

    # 1. SMART DECISION: Calculate District Coordinate Means
    district_coords = sample_df.groupby('district')[['latitude', 'longitude']].mean().to_dict('index')

    # 2. TRIGGER: Place Selectbox OUTSIDE the form for instant updates
    # This causes a rerun every time the selection changes
    dist = st.selectbox("Select Target District", options=encoder.classes_)

    # 3. SET DEFAULTS: Get the mean values immediately based on selection
    default_lat = district_coords[dist]['latitude']
    default_lon = district_coords[dist]['longitude']

    # 4. THE FORM: Now we wrap the actual analysis inputs
    with st.form("predictor_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            size = st.slider("Household Size", 1, 15, 5)
            elev = st.number_input("Elevation (m)", 1400, 3000, 1800)
            # This now updates instantly when the selectbox above changes
            lat = st.number_input("Latitude (Approx)", value=default_lat, format="%.6f")
            
        with col_b:
            dist_market = st.number_input("Distance to Market (km)", 0.0, 30.0, 5.0)
            base_fuel = st.number_input("Baseline Fuel (kg/person/week)", 1.0, 30.0, 8.0)
            # This now updates instantly when the selectbox above changes
            lon = st.number_input("Longitude (Approx)", value=default_lon, format="%.6f")
            
            date = st.date_input("Planned Distribution Date")

        submit = st.form_submit_button("Analyze Household Risk")

    if submit:
        # Prediction logic remains the same...
        input_raw = pd.DataFrame([{
            'district': dist, 'latitude': lat, 'longitude': lon,
            'household_size': size, 'elevation_m': elev,
            'distance_to_market_km': dist_market,
            'baseline_fuel_kg_person_week': base_fuel,
            'distribution_date': date.strftime("%d/%m/%Y")
        }])

        processed = utils.full_pipeline(input_raw)
        X_input = processed[meta['features']]
        X_input['district'] = encoder.transform(X_input['district'])
        prob = model.predict_proba(X_input)[0][1]
        
    
        # Gauge and Recommendation code
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
             # Gauge code here...
             fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob * 100,
                title = {'text': "Risk Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'steps' : [
                        {'range': [0, 30], 'color': "#2ecc71"},
                        {'range': [30, 70], 'color': "#f1c40f"},
                        {'range': [70, 100], 'color': "#e74c3c"}]}))
             st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.subheader("Management Recommendation")
            if prob > 0.70:
                st.error(f"🚨 **CRITICAL RISK ({prob*100:.1f}%)**")
                st.write("Household is geographically isolated. **Action:** Priority visit required.")
            elif prob > 0.30:
                st.warning(f"⚠️ **MODERATE RISK ({prob*100:.1f}%)**")
                st.write("Monitor usage via SMS alerts.")
            else:
                st.success(f"✅ **LOW RISK ({prob*100:.1f}%)**")
                st.write("Standard monitoring applies.")


# 6. PAGE: BATCH PREDICTION
elif page == "Batch Intelligence":
    st.title("📂 Batch Intelligence")

    # Add this "Help" section
    with st.expander("Don't have a CSV to test?"):
        sample_file = pd.read_csv('data/test_households_sample.csv')
        st.download_button("📥 Download Test Template", 
                           sample_file.to_csv(index=False).encode('utf-8'), 
                           "template.csv", "text/csv")
        
    st.markdown("Upload a CSV file containing new household distributions to identify priority zones.")
    
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        data_in = pd.read_csv(file)
        with st.spinner("Processing geographic data..."):
            # Apply full pipeline (Cleaning + Engineering)
            processed_data = utils.full_pipeline(data_in)
            
            # Prepare for model
            X_batch = processed_data[meta['features']].copy()
            X_batch['district'] = encoder.transform(X_batch['district'])
            
            # Get Probabilities
            processed_data['Adoption_Risk_Prob'] = model.predict_proba(X_batch)[:, 1]
            processed_data['Priority_Level'] = pd.cut(processed_data['Adoption_Risk_Prob'], 
                                                     bins=[0, 0.3, 0.7, 1.0], 
                                                     labels=['Low', 'Medium', 'High'])
            
            st.subheader("Risk Assessment Results")
            st.dataframe(processed_data[['household_id', 'district', 'Adoption_Risk_Prob', 'Priority_Level']].sort_values('Adoption_Risk_Prob', ascending=False))
            
            # Download button
            csv = processed_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Field Priority List", csv, "field_priority_list.csv", "text/csv")

# 7. PAGE: PERFORMANCE
elif page == "Model Performance":
    st.title("⚙️ Model Technical Validation")
    st.json(meta)
    st.info("This model uses HistGradientBoosting which natively handles missing values and uses a 'balanced' class weight to adjust for low adoption frequency.")


# Place this at the very bottom of app.py, outside of any column or IF block
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(14, 17, 23, 0.8);
        color: #9ea0a5;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 100;
    }
    </style>
    <div class="footer">
        Built by Mr. Emmanuel Ansah as part of the DelAgua Data Science Assessment • February 2026
    </div>
    """, unsafe_allow_html=True)