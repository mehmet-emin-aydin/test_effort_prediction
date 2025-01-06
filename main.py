import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# MMRE Hesaplama
def calculate_mmre(y_true, y_pred):
    epsilon = 1e-6
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))

# Model Eğitimi ve Değerlendirme
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mmre = calculate_mmre(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, mmre, rmse

# Veri Yükleme
@st.cache_data
def load_data():
    data = pd.read_csv("datasets/coc81.csv", delimiter=",")
    data.columns = data.columns.str.strip().str.replace('[$?]', '', regex=True)
    if 'project_id' in data.columns:
        data.drop('project_id', axis=1, inplace=True)
    return data

# Streamlit Başlığı
st.title("Model Training and Evaluation")
st.sidebar.header("Hiperparametreler")

# Veriyi Yükleme
data = load_data()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data Description Statistics (Describe)
st.subheader("Data Description Statistics")
st.write(data.describe())
corr_data = data.drop("dev_mode", axis=1)

# Korelasyon Matrisini Görselleştirme
st.subheader("Correlation Matrix")
corr_matrix = corr_data.corr()

# Korelasyon Matrisini Isı Haritası Olarak Göster
fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and axis explicitly
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)  # Pass the figure explicitly to Streamlit

# Özellikler ve Hedef Değişken
categorical_columns = data.select_dtypes(include=['object']).columns
numerical_columns = data.select_dtypes(include=[np.number]).columns
X = data[numerical_columns].drop('<actual', axis=1)
y = data['<actual']

# Kategorik veriler için One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
dev_mode_encoded = encoder.fit_transform(data[['dev_mode']])
dev_mode_encoded_df = pd.DataFrame(dev_mode_encoded, columns=encoder.get_feature_names_out(['dev_mode']))

# Sayısal Veriler için Ölçekleme
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled = pd.concat([X_scaled, dev_mode_encoded_df], axis=1)

# Eğitim ve Test Bölme
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hiperparametre Ayarları
st.sidebar.subheader("SVR Hyperparameters")
svr_C = st.sidebar.slider("C (SVR)", 0.1, 10.0, 1.0, step=0.1)
svr_e = st.sidebar.slider("e (epsilon)", 0.001, 1.0, 0.01, step=0.001)
svr_gamma = st.sidebar.selectbox("Gamma (SVR)", ["scale", "auto"])

st.sidebar.subheader("ANN Hyperparameters")
ann_hidden_layer_sizes = st.sidebar.selectbox("Hidden Layers (ANN)", [(50,), (100,), (50, 50), (50,50,50)])
ann_max_iter = st.sidebar.slider("Max Iterations (ANN)", 100, 100000, 500, step=100)

st.sidebar.subheader("DecisionTree hyperparameters")
dt_max_depth = st.sidebar.slider("Max Depth (DecisionTree)", 1, 1000, None, step=1)
dt_min_samples_split = st.sidebar.slider("Min Samples Split (DecisionTree)", 2, 100, 2, step=1)

# Eğitim Butonu
if st.button("train and evaluate models"):
    results = {}

    # SVR Modeli
    svr_model = SVR(kernel='rbf', epsilon=svr_e, C=svr_C, gamma=svr_gamma)
    svr_y_pred, svr_mmre, svr_rmse = train_and_evaluate_model(svr_model, X_train, y_train, X_test, y_test)
    results["SVR"] = {"MMRE": svr_mmre, "RMSE": svr_rmse}

    # ANN Modeli
    ann_model = MLPRegressor(hidden_layer_sizes=ann_hidden_layer_sizes, max_iter=ann_max_iter, random_state=42)
    ann_y_pred, ann_mmre, ann_rmse = train_and_evaluate_model(ann_model, X_train, y_train, X_test, y_test)
    results["ANN"] = {"MMRE": ann_mmre, "RMSE": ann_rmse}

    # DecisionTree Modeli
    dt_model = DecisionTreeRegressor(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, random_state=42)
    dt_y_pred, dt_mmre, dt_rmse = train_and_evaluate_model(dt_model, X_train, y_train, X_test, y_test)
    results["DecisionTree"] = {"MMRE": dt_mmre, "RMSE": dt_rmse}

    # Sonuçları Görüntüleme
    results_df = pd.DataFrame(results).T

    # Plot Table
    st.write("Training Results (Plot Table):")
    st.table(results_df.style.format({"MMRE": "{:.4f}", "RMSE": "{:.4f}"}))

    # Scatter Plot for Original vs Predicted
    # Scatter Plot for SVR: Original vs Predicted
    st.subheader("Original vs Predicted Scatter Plot (SVR)")
    fig_svr, ax_svr = plt.subplots(figsize=(8, 6))
    ax_svr.scatter(y_test, svr_y_pred, color='blue', alpha=0.5, label='SVR Predictions')
    ax_svr.scatter(y_test, y_test, color='black', marker='x', label='Actual y_test', s=50)  # Actual y_test points
    ax_svr.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax_svr.set_xlabel("Original")
    ax_svr.set_ylabel("Predicted")
    ax_svr.set_title("SVR: Original vs Predicted")
    ax_svr.legend()
    st.pyplot(fig_svr)

    # Scatter Plot for ANN: Original vs Predicted
    st.subheader("Original vs Predicted Scatter Plot (ANN)")
    fig_ann, ax_ann = plt.subplots(figsize=(8, 6))
    ax_ann.scatter(y_test, ann_y_pred, color='green', alpha=0.5, label='ANN Predictions')
    ax_ann.scatter(y_test, y_test, color='black', marker='x', label='Actual y_test', s=50)  # Actual y_test points
    ax_ann.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax_ann.set_xlabel("Original")
    ax_ann.set_ylabel("Predicted")
    ax_ann.set_title("ANN: Original vs Predicted")
    ax_ann.legend()
    st.pyplot(fig_ann)

    # Scatter Plot for DecisionTree: Original vs Predicted
    st.subheader("Original vs Predicted Scatter Plot (DecisionTree)")
    fig_dt, ax_dt = plt.subplots(figsize=(8, 6))
    ax_dt.scatter(y_test, dt_y_pred, color='orange', alpha=0.5, label='DecisionTree Predictions')
    ax_dt.scatter(y_test, y_test, color='black', marker='x', label='Actual y_test', s=50)  # Actual y_test points
    ax_dt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax_dt.set_xlabel("Original")
    ax_dt.set_ylabel("Predicted")
    ax_dt.set_title("DecisionTree: Original vs Predicted")
    ax_dt.legend()
    st.pyplot(fig_dt)

