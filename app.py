import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.title("US Market Trends - Predictive Analysis")

@st.cache_data
def load_data():
    data = pd.read_csv("us_market_data_hourly_with_inflation_modified.zip")
    data.columns = data.columns.str.strip()
    if "observation_date" in data.columns:
        data["observation_date"] = pd.to_datetime(data["observation_date"], format="%d-%m-%Y %H:%M", errors='coerce')
    return data

data = load_data()

st.write("Debug: Data loaded successfully")

st.write("### Displaying Data")
st.write(data.head())

st.write("### Data Overview")
st.write(data.dtypes)  # Check column types

# Correlation Heatmap
st.title("US Market Data Correlation Heatmap")
st.sidebar.header("Heatmap Settings")
columns = st.sidebar.multiselect("Select Columns for Correlation", options=data.columns.drop("observation_date"), default=data.columns.drop("observation_date"))

if len(columns) > 1:
    correlation_matrix = data[columns].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.warning("Please select at least two columns for the heatmap.")

# Graph Plotting
st.sidebar.subheader("Data Visualization")
if st.sidebar.checkbox("Plot Data Columns"):
    column = st.sidebar.selectbox("Select Column to Plot", data.columns[1:])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["observation_date"], data[column], label=column, color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.set_title(f"{column} Over Time")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# Model Training Configuration
st.sidebar.header("Model Training Options")
train_size = st.sidebar.slider("Training Dataset Size (%)", 10, 90, 80)
model_choice = st.sidebar.radio("Select Model", ["Random Forest", "XGBoost", "LSTM", "Neural Network", "Support Vector Regression"])

if st.sidebar.button("Train Model"):
    st.write("### Model Training and Evaluation")
    
    features = data.drop(columns=["observation_date", "Inflation"], errors='ignore')
    target = data["Inflation"]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, train_size=train_size/100, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    history = None  # For LSTM and Neural Network models

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    elif model_choice == "XGBoost":
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
    elif model_choice == "LSTM":
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    elif model_choice == "Neural Network":
        model = Sequential([
            Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    else:
        model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Display Metrics in a Table
    st.write("### Model Performance Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R²"],
        "Validation Score": [val_mae, val_mse, val_rmse, val_r2],
        "Test Score": [test_mae, test_mse, test_rmse, test_r2]
    })
    st.table(metrics_df)

    # Plot Actual vs Predicted
    st.write("### Model Predictions: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values, label="Actual", color="blue")
    ax.plot(y_test_pred, label="Predicted", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("Inflation")
    ax.set_title(f"{model_choice}: Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

    # Plot Training Loss Curve for LSTM & Neural Network
    if history:
        st.write("### Model Training Loss")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history.history["loss"], label="Training Loss", color="blue")
        ax.plot(history.history["val_loss"], label="Validation Loss", color="red")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss Over Epochs")
        ax.legend()        
        st.pyplot(fig)
