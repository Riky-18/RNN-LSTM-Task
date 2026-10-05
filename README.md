# RNN & LSTM Time-Series Forecasting

This project explores how recurrent neural networks can learn patterns in time-series data. Its main example uses daily minimum temperatures recorded in Melbourne, Australia, to predict future temperatures from recent observations.

The project includes a notebook for exploring and training the models, saved model files, and a Streamlit app for trying out forecasts without retraining.

## What you can do

- Explore the temperature dataset and its history.
- Compare a simple persistence baseline with a vanilla RNN and an LSTM.
- Review the saved evaluation metrics and training curves.
- Forecast temperatures from a historical date window or a sequence you enter.
- Switch between the saved models and request forecasts up to seven days ahead.

## Results

The table below reports the values saved in [`preprocessing/results.json`](preprocessing/results.json). On that recorded test evaluation, the vanilla RNN performed slightly better than the LSTM. Results can vary if the data preparation or training process changes.

| Model | MAE (°C) | RMSE (°C) | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Persistence baseline | 2.04 | 2.57 | 0.565 | 22.35% |
| Vanilla RNN | 1.74 | 2.24 | 0.671 | 19.21% |
| LSTM | 1.75 | 2.24 | 0.669 | 19.65% |

## Project contents

```text
.
├── app.py
├── requirements.txt
├── dataset/
│   ├── cleaned_temperatures.csv
│   └── daily-minimum-temperatures-in-me.csv
├── models/
│   ├── best_model.keras
│   ├── lstm_model.keras
│   └── rnn_model.keras
├── notebooks/
│   └── RNN_LSTM_Project.ipynb
├── preprocessing/
│   ├── metadata.json
│   ├── results.json
│   └── scaler.pkl
├── daily-minimum-temperatures-in-me.csv
├── Electric_Production.csv
├── monthly-beer-production-in-austr.csv
└── sales-of-shampoo-over-a-three-ye.csv
```

The Streamlit app uses the Melbourne daily-temperature dataset in `dataset/` (and can also find the copy in the project root). The other CSV files are included as additional time-series data; they are not used by the app's temperature forecasting workflow.

## Run it locally

Use Python with the packages listed in `requirements.txt`.

1. Create and activate a virtual environment:

   **Windows (PowerShell)**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the app from the project folder:

   ```bash
   streamlit run app.py
   ```

   Open the local URL printed in the terminal (usually `http://localhost:8501`).

To explore the training and analysis workflow, open `notebooks/RNN_LSTM_Project.ipynb` in Jupyter Notebook or Google Colab.
