import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Set Page Config
st.set_page_config(
    page_title="RNN & LSTM Daily Temperature Forecasting",
    layout="wide"
)

# Function to load resources cleanly with caching
@st.cache_resource
def load_resources():
    import tensorflow as tf
    scaler_path = 'preprocessing/scaler.pkl'
    meta_path = 'preprocessing/metadata.json'
    results_path = 'preprocessing/results.json'
    
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            
    models = {}
    if os.path.exists('models/best_model.keras'):
        models['Best Model'] = tf.keras.models.load_model('models/best_model.keras')
    if os.path.exists('models/lstm_model.keras'):
        models['LSTM'] = tf.keras.models.load_model('models/lstm_model.keras')
    if os.path.exists('models/rnn_model.keras'):
        models['Vanilla RNN'] = tf.keras.models.load_model('models/rnn_model.keras')
        
    return scaler, metadata, results, models

@st.cache_data
def load_dataset():
    paths = ['dataset/daily-minimum-temperatures-in-me.csv', 'daily-minimum-temperatures-in-me.csv']
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [col.strip() for col in df.columns]
            temp_col = [c for c in df.columns if 'temp' in c.lower() or 'temperature' in c.lower()][0]
            date_col = [c for c in df.columns if 'date' in c.lower()][0]
            df[temp_col] = df[temp_col].astype(str).str.replace('?', '', regex=False)
            df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, temp_col]).sort_values(date_col).reset_index(drop=True)
            return df, date_col, temp_col
    return None, None, None

def main():
    st.title("RNN & LSTM Daily Minimum Temperature Forecasting")
    st.markdown("""
    An end-to-end Deep Learning application using **Vanilla Recurrent Neural Networks (RNN)** 
    and **Long Short-Term Memory (LSTM)** architectures to forecast daily minimum temperatures 
    in Melbourne over time.
    """)
    
    scaler, metadata, results, models = load_resources()
    df, date_col, temp_col = load_dataset()
    
    if df is None:
        st.error("Dataset file not found! Please check `dataset/daily-minimum-temperatures-in-me.csv`.")
        return

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to Section:", [
        "Overview & Dataset",
        "Model Performance",
        "Interactive Forecasting",
        "Error & Residual Analysis"
    ])
    
    lookback = metadata.get("lookback", 30)
    
    if page == "Overview & Dataset":
        st.header("Dataset & Problem Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Date Range", f"{df[date_col].dt.year.min()} - {df[date_col].dt.year.max()}")
        col3.metric("Min Temp (°C)", f"{df[temp_col].min():.1f}°C")
        col4.metric("Max Temp (°C)", f"{df[temp_col].max():.1f}°C")
        
        st.subheader("Raw Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Historical Minimum Temperature Sequence")
        st.line_chart(df.set_index(date_col)[temp_col])
        
        st.markdown("""
        ### Final Problem Formulation:
        - **Task**: Univariate Time-Series Forecasting (Regression)
        - **Target Variable**: Daily Minimum Temperature (°C)
        - **Sequence Window (Lookback)**: 30 consecutive days of historical temperatures
        - **Objective**: Predict temperature at day $t$ given values from $t-30$ to $t-1$.
        """)

    elif page == "Model Performance":
        st.header("Model Architectures & Benchmark Performance")
        
        if results:
            st.subheader("Model Comparison Metrics (Evaluated on Untouched Test Set)")
            
            res_df = pd.DataFrame([
                {"Model": "Baseline (Persistence)", **results.get("Baseline", {})},
                {"Model": "Vanilla RNN", **results.get("Vanilla RNN", {})},
                {"Model": "LSTM (Best Architecture)", **results.get("LSTM", {})}
            ]).set_index("Model")
            
            st.dataframe(res_df.style.highlight_min(axis=0, color='lightgreen', subset=['MAE', 'MSE', 'RMSE', 'MAPE'])
                                   .highlight_max(axis=0, color='lightgreen', subset=['R2']),
                        use_container_width=True)
            
            st.success(f"Best Performing Model: **{results.get('Best_Model', 'LSTM')}**")
            
            st.subheader("Training vs Validation Loss Curves")
            c1, c2 = st.columns(2)
            
            if 'RNN_history' in results:
                with c1:
                    st.markdown("**Vanilla RNN Training Curves**")
                    rnn_hist = pd.DataFrame(results['RNN_history'])
                    st.line_chart(rnn_hist[['loss', 'val_loss']])
                    
            if 'LSTM_history' in results:
                with c2:
                    st.markdown("**LSTM Training Curves**")
                    lstm_hist = pd.DataFrame(results['LSTM_history'])
                    st.line_chart(lstm_hist[['loss', 'val_loss']])
        else:
            st.info("Train models first to view performance curves!")

    elif page == "Interactive Forecasting":
        st.header("Interactive Temperature Prediction")
        
        selected_model_name = st.selectbox("Select Trained Model for Inference:", list(models.keys()))
        model = models.get(selected_model_name)
        
        if model is None or scaler is None:
            st.error("Model or Scaler object not found!")
            return
            
        st.subheader("Select Input Sequence Mode")
        mode = st.radio("Choose Input Mode:", ["Historical Date Window", "Custom Temperature Sequence Entry"])
        
        if mode == "Historical Date Window":
            max_start_idx = len(df) - lookback - 7
            start_date_idx = st.slider(
                "Select Starting Index in Dataset for 30-Day History Window:",
                0, max_start_idx, len(df) - lookback - 30
            )
            
            history_df = df.iloc[start_date_idx : start_date_idx + lookback]
            actual_future_df = df.iloc[start_date_idx + lookback : start_date_idx + lookback + 7]
            
            st.write(f"**Selected 30-Day Window**: {history_df[date_col].dt.date.min()} to {history_df[date_col].dt.date.max()}")
            st.line_chart(history_df.set_index(date_col)[temp_col])
            
            input_temps = history_df[temp_col].values.reshape(-1, 1)
            
        else:
            st.markdown("Enter 30 daily temperature values (°C) comma-separated or use the slider default:")
            default_str = ", ".join([str(round(x, 1)) for x in df[temp_col].tail(30).tolist()])
            user_input_str = st.text_area("30-Day Sequence (°C):", default_str)
            
            try:
                input_temps = np.array([float(x.strip()) for x in user_input_str.split(',')]).reshape(-1, 1)
                if len(input_temps) != lookback:
                    st.warning(f"Expected exactly {lookback} temperature values, got {len(input_temps)}. Padding/truncating automatically.")
                    if len(input_temps) < lookback:
                        input_temps = np.pad(input_temps, ((lookback - len(input_temps), 0), (0, 0)), 'edge')
                    else:
                        input_temps = input_temps[-lookback:]
            except Exception as e:
                st.error(f"Invalid input format: {e}")
                return

        forecast_horizon = st.slider("Select Forecast Horizon (Days ahead):", 1, 7, 1)
        
        if st.button("Forecast Temperature", type="primary"):
            curr_sequence = scaler.transform(input_temps).copy()
            preds_scaled = []
            
            for _ in range(forecast_horizon):
                X_in = curr_sequence[-lookback:].reshape(1, lookback, 1)
                pred_scaled = model.predict(X_in, verbose=0)[0, 0]
                preds_scaled.append(pred_scaled)
                curr_sequence = np.vstack([curr_sequence, [[pred_scaled]]])
                
            preds_orig = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            
            st.subheader("Forecast Results")
            for i, p in enumerate(preds_orig, 1):
                st.metric(f"Day +{i} Predicted Minimum Temp", f"{p:.2f} °C")
                
            # Chart predictions
            hist_vals = input_temps.flatten()
            combined_vals = np.concatenate([hist_vals, preds_orig])
            idx_labels = [f"Day -{lookback - i}" for i in range(lookback)] + [f"Forecast +{i}" for i in range(1, forecast_horizon + 1)]
            
            chart_df = pd.DataFrame({"Temperature (°C)": combined_vals}, index=idx_labels)
            st.line_chart(chart_df)

    elif page == "Error & Residual Analysis":
        st.header("Residual & Error Diagnostics")
        st.markdown("""
        Detailed residual inspection on the test dataset to verify forecast bias and error distributions.
        """)
        if os.path.exists('preprocessing/results.json'):
            with open('preprocessing/results.json') as f:
                res = json.load(f)
            
            st.json(res)
        else:
            st.info("Results file unavailable.")

if __name__ == '__main__':
    main()
