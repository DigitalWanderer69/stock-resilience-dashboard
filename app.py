import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# --- Page Configuration ---
st.set_page_config(page_title="S&P 500 Resilience Dashboard", layout="wide")

# --- Load Models & Data (Cached for speed) ---
@st.cache_resource
def load_models():
    # Loading the single Champion models that handle everything
    xgb_mdd = joblib.load('champion_xgb_mdd.pkl')
    rf_ttr = joblib.load('champion_rf_ttr.pkl')
    features = joblib.load('model_features.pkl')
    return xgb_mdd, rf_ttr, features

@st.cache_data
def load_all_datasets():
    # Load the master file
    master_df = pd.read_csv('sp500_dashboard_light.csv')
    
    # Fix column naming discrepancy (replace space with underscore)
    master_df = master_df.rename(columns={'Adj Close': 'Adj_Close'})
    
    # Split the data into a dictionary for the dashboard tabs/filters
    return {
        "COVID-19 Crash (2020)": master_df[master_df['Crisis_Type'] == 'COVID-19 Crash'],
        "Global Financial Crisis (2008)": master_df[master_df['Crisis_Type'] == 'Global Financial Crisis'],
        "Dot-Com Bubble (2000)": master_df[master_df['Crisis_Type'] == 'Dot-Com Bubble']
    }

xgb_model, rf_model, feature_cols = load_models()
datasets = load_all_datasets()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Historical Benchmarking", "Predictive Simulator"])

# --- PAGE 1: Historical Benchmarking ---
if page == "Historical Benchmarking":
    st.title("Historical Market Benchmarking")
    st.write("Explore sector resilience across past market crashes.")
    
    st.subheader("1. Classic Price Trajectory")
    filter_col, chart_col = st.columns([1, 3])
    
    with filter_col:
        selected_crisis = st.selectbox("Select Historical Crisis:", list(datasets.keys()))
        df_selected = datasets[selected_crisis] 
        selected_sector = st.selectbox("Select GICS Sector:", df_selected['Sector'].unique())
        st.write(f"Tracking the daily average Adjusted Close price for the **{selected_sector}** sector during the **{selected_crisis}**.")
        
    with chart_col:
        sector_data = df_selected[df_selected['Sector'] == selected_sector].copy()
        sector_data['Date'] = pd.to_datetime(sector_data['Date'])
        sector_data = sector_data.sort_values('Date')
        daily_avg = sector_data.groupby('Date')['Adj_Close'].mean().reset_index()
        daily_avg.set_index('Date', inplace=True)
        st.line_chart(daily_avg['Adj_Close'])
        st.caption(" **How to read this:** This shows the actual historical price path. Look for the lowest point (the trough) to see the Maximum Drawdown, and how long it took to climb back to the start.")
        
    st.divider() 

    st.subheader("2. Risk vs. Reward Landscape")
    risk_reward_df = df_selected.groupby('Sector').agg(
        Volatility=('Daily_Return', 'std'),
        Mean_Return=('Daily_Return', 'mean')
    ).reset_index()
    
    st.scatter_chart(
        data=risk_reward_df, x='Volatility', y='Mean_Return', color='Sector', size=200, use_container_width=True
    )
    st.caption(" **How to read this:** Sectors higher up on the Y-axis have better growth. Sectors further to the right on the X-axis are more dangerous/volatile. The ideal safe sector is in the top-left corner.")

    st.subheader("3. Sector Volatility Ranking")
    vol_df = df_selected.groupby('Sector')['Daily_Return'].std().sort_values(ascending=False)
    st.bar_chart(vol_df)
    st.caption(" **How to read this:** The tallest bars represent the most chaotic and risky sectors.")

    st.subheader("4. Sector Mean Return Ranking")
    mean_return_df = df_selected.groupby('Sector')['Daily_Return'].mean().sort_values(ascending=False)
    st.bar_chart(mean_return_df)
    st.caption(" **How to read this:** The tallest bars represent the sectors that actually grew the most (or lost the least) on a day-to-day average.")

    st.subheader(f"Dataset Overview: {selected_crisis}")
    st.dataframe(df_selected.head(10))

# --- PAGE 2: Predictive Simulator ---
elif page == "Predictive Simulator":
    st.title(" Predictive Crash Simulator")
    st.write("Set hypothetical market conditions and macroeconomic environments to forecast sector resilience.")
    
    col1, col2 = st.columns([1, 2])
    base_df = datasets["COVID-19 Crash (2020)"]
    
    with col1:
        st.subheader("Set Market Parameters")
        
        # User sets the macroeconomic environment flag
        model_context = st.selectbox(
            "Macroeconomic Environment (Crisis Context):", 
            ["COVID-19 (Pandemic Shock)", "2008 GFC (Systemic Banking Shock)", "2000 Dot-Com (Asset Bubble)"]
        )
        
        valid_sectors = [
            "Communication Services", 
            "Consumer Cyclical", 
            "Consumer Defensive", 
            "Energy", 
            "Financial Services", 
            "Healthcare", 
            "Industrials", 
            "Real Estate", 
            "Technology", 
            "Utilities"
        ]

        input_sector = st.selectbox("Select GICS Sector", valid_sectors)
        input_vol = st.slider("Baseline Volatility (%)", 1.0, 8.0, 2.6, step=0.1)
        input_mean_return = st.slider("Mean Daily Return", -0.05, 0.05, 0.001, step=0.001)
        
    with col2:
        st.subheader("Simulation Results")
        if st.button("Run Simulation", type="primary"):
            
            # --- 1. Prepare Inputs for the Champion Model ---
            input_data = pd.DataFrame(columns=feature_cols)
            input_data.loc[0] = 0  # Fill all with 0 initially
            
            # Convert numeric columns to floats and multiply by 100 to match training scale
            input_data['Volatility'] = input_data['Volatility'].astype(float)
            input_data['Mean_Return'] = input_data['Mean_Return'].astype(float)

            input_data.at[0, 'Volatility'] = input_vol
            input_data.at[0, 'Mean_Return'] = input_mean_return
            
            # Set Sector dummy
            sector_col = f"Sector_{input_sector}"
            if sector_col in input_data.columns:
                input_data.at[0, sector_col] = 1
                
            # --- 1B. Explicit Crisis Type Mapping ---
            # Ensure crisis columns are reset to 0 first
            if "Crisis_Type_Dot-Com Bubble" in input_data.columns:
                input_data.at[0, "Crisis_Type_Dot-Com Bubble"] = 0
            if "Crisis_Type_Global Financial Crisis" in input_data.columns:
                input_data.at[0, "Crisis_Type_Global Financial Crisis"] = 0
                
            # Explicitly turn on the selected crisis
            if "2000 Dot-Com" in model_context:
                if "Crisis_Type_Dot-Com Bubble" in input_data.columns:
                    input_data.at[0, "Crisis_Type_Dot-Com Bubble"] = 1
            elif "2008 GFC" in model_context:
                if "Crisis_Type_Global Financial Crisis" in input_data.columns:
                    input_data.at[0, "Crisis_Type_Global Financial Crisis"] = 1
            elif "COVID-19" in model_context:
                # COVID-19 is the reference category (both dummy columns remain 0)
                pass
                
            # --- 2. Generate Predictions ---
            # XGBoost predicts Max Drawdown
            pred_mdd = float(xgb_model.predict(input_data)[0])
            display_mdd = f"{pred_mdd * 100:.2f}%"
            crash_days = 45
            
            # Random Forest predicts Time-to-Recovery (revert log transform)
            pred_ttr_log = float(rf_model.predict(input_data)[0])
            sim_recovery_days = int(round(np.expm1(pred_ttr_log)))
            
            # Set Classification based on Random Forest TTR prediction
            if sim_recovery_days >= 9999 or pred_mdd <= -0.95:
                ttr_value = "No Recovery"
                ttr_status = "Delisting Risk"
            elif sim_recovery_days > 365:
                ttr_value = f"{sim_recovery_days:,} Days"
                ttr_status = "Slow Recovery"
            else:
                ttr_value = f"{sim_recovery_days:,} Days"
                ttr_status = "Fast Recovery"
                
            total_days = crash_days + sim_recovery_days
            
            st.info(f"Predictions powered by **{model_context}** Macro-Dynamics")
            
            # 3-Column KPI Display
            met1, met2, met3 = st.columns([1, 1, 1.4])
            met1.metric(label="Max Drawdown", value=display_mdd)
            met2.metric(label="Recovery Time", value=ttr_value)
            met3.metric(label="Recovery Classification", value=ttr_status)
            
            # --- 3. Generate Visual Trajectory ---
            st.subheader("Projected Crash & Recovery Path")
            
            # Cap the graph view at 410 days to keep long recoveries legible
            chart_cap = min(total_days, 410)
            
            full_index = np.arange(chart_cap + 1)
            trend_line = np.zeros(chart_cap + 1)
            
            # Plot the drop
            mdd_value = pred_mdd * 100 
            drop_slope = mdd_value / crash_days
            trend_line[:crash_days+1] = 100 + (np.arange(crash_days+1) * drop_slope)
            
            # Plot the recovery 
            recovery_slope = abs(mdd_value) / sim_recovery_days
            trend_line[crash_days:] = trend_line[crash_days] + (np.arange(chart_cap - crash_days + 1) * recovery_slope)
            
            # Add Volatility Noise
            daily_noise = np.random.normal(0, input_vol, size=chart_cap + 1)
            daily_noise[0] = 0 
            simulated_path = np.clip(trend_line + daily_noise, a_min=0, a_max=None)
            
            # --- ALTAIR VISUALIZATION ---
            chart_df = pd.DataFrame({
                'Day': full_index,
                'Portfolio Value (%)': simulated_path,
                'Phase': ['Crash Phase' if i <= crash_days else 'Recovery Phase' for i in full_index]
            })
            
            # Line chart
            line = alt.Chart(chart_df).mark_line(size=3).encode(
                x=alt.X('Day', title='Days Since Crash Started', scale=alt.Scale(domain=[0, chart_cap])),
                y=alt.Y('Portfolio Value (%)', scale=alt.Scale(domain=[min(simulated_path)-5, 105])),
                color=alt.Color('Phase', 
                                scale=alt.Scale(domain=['Crash Phase', 'Recovery Phase'], range=['#ff4b4b', '#00cc96']), 
                                legend=alt.Legend(title="Market Cycle", orient="bottom"))
            )
            
            # Key milestone markers
            if ttr_status == "Fast Recovery":
                key_days = [0, crash_days, total_days]
                key_labels = ['Start', 'Trough (MDD)', f'Recovered (Day {total_days:,})']
            else:
                key_days = [0, crash_days, chart_cap]
                if pred_mdd <= -0.95:
                    key_labels = ['Start', 'Trough (MDD)', 'Severe Delisting Risk']
                elif pred_mdd <= -0.70:
                    key_labels = ['Start', 'Trough (MDD)', f'Stagnation (Day {sim_recovery_days:,})']
                else:
                    key_labels = ['Start', 'Trough (MDD)', f'Slow Recovery (Day {sim_recovery_days:,})']
                
            key_points = pd.DataFrame({
                'Day': key_days,
                'Portfolio Value (%)': [simulated_path[i] for i in key_days],
                'Label': key_labels
            })
            
            points = alt.Chart(key_points).mark_point(size=120, filled=True, color='white', opacity=1).encode(
                x='Day',
                y='Portfolio Value (%)',
                tooltip=['Label', 'Day', 'Portfolio Value (%)']
            )
            
            text = points.mark_text(
                align='left',
                baseline='middle',
                dx=12,
                dy=-12,
                fontSize=14,
                fontWeight='bold',
                color='white' 
            ).encode(
                text='Label'
            )
            
            st.altair_chart(line + points + text, use_container_width=True)
                
            st.caption(" **How to interpret this simulation:** The **red line** represents the panic drop down to the **Predicted Maximum Drawdown** (Trough). The **green line** shows the volatile climb back to baseline dictated by your **Predicted Time-to-Recovery**. The jagged movements represent standard market noise based on your **Volatility** setting.")
