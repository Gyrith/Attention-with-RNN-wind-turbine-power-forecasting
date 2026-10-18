# Attention with RNN: Wind Turbine Power Forecasting

Forecasting wind turbine power output with an attention-enhanced LSTM in TensorFlow/Keras. The model predicts the turbine's next 10-minute active power (kW) from the previous 24 readings (4 hours) and uses a custom attention layer to show which time steps it relies on.
## Dataset

`T1.csv`: 50,530 readings at 10-minute intervals covering all of 2018.

## Results

| Metric | Value |
|---|---|
| Test loss (scaled MSE) | 0.0040 |
| MSE | 52,584 kW² |
| RMSE | 229.3 kW |
| R² | 0.971 |

- RMSE is about 6% of the turbine's ~3,600 kW maximum output.
- The attention weights rise steadily toward the most recent time steps. The last ~5 steps (about 50 minutes) carry most of the weight, so the model relies mainly on recent conditions.
- Main weaknesses: a slight over-prediction when the turbine is idle (about 20-50 kW when actual output is 0) and a small lag at sudden changes.

| Column | Used as |
|---|---|
| `Date/Time` | Converted to datetime, used to sort the data |
| `LV ActivePower (kW)` | Feature and prediction target |
| `Wind Speed (m/s)` | Feature |
| `Theoretical_Power_Curve (KWh)` | Feature |
| `Wind Direction (°)` | Not used |

No values were missing. The code still checks for missing values and fills small gaps by linear interpolation.

## Method

1. **Prepare the data:** convert dates, check missing values, explore with summary statistics and scatter plots, and scale the features with `MinMaxScaler` (fit on the training period only to avoid leakage).
2. **Create sequences:** 24-step windows of all three features, with the next `LV ActivePower` value as the target. `X` is (50506, 24, 3) and `y` is (50506,).
3. **Split:** chronological 80/20, with no shuffling. That gives 40,404 training and 10,102 test samples. The test set is roughly the last two months of 2018.
4. **Model:** Input, then LSTM(64) and Dropout(0.2), then LSTM(32) and Dropout(0.2), then a custom additive attention layer, then Dense(1).
5. **Train:** Adam optimizer, MSE loss, up to 30 epochs, batch size 64, 10% validation split, and early stopping on validation loss (patience 5).
6. **Evaluate:** inverse-transform predictions to kW, then compute MSE, RMSE and R². Plot actual vs. predicted values (time series and scatter) and the attention weights.

## Project Structure
```
├── C10_M05_Lab.ipynb   # Notebook with the full solution and outputs
├── T1.csv              # Wind turbine dataset
└── README.md
```
Python, NumPy, pandas, Matplotlib, scikit-learn, TensorFlow/Keras.