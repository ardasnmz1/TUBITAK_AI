import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import os

# Veri yükleme ve ön işleme 
def load_and_preprocess_data(data_path):
    # Veri setini yükle
    df = pd.read_csv(data_path)
    
    # Eksik değerleri doldur
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Özellikler ve hedef değişkeni ayır
    X = df_imputed.drop('target', axis=1)
    y = df_imputed['target']
    
    # Veriyi ölçeklendir
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Eğitim ve test setlerini ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def train_xgboost_model(X_train, y_train):
    # XGBoost modelini oluştur ve eğit
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def create_mc_dropout_model(input_dim):
    # Monte Carlo Dropout modelini oluştur
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=Huber()
    )
    
    return model

def train_mc_dropout_model(model, X_train, y_train, X_test, y_test):
    # Early stopping için callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Modeli eğit
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def main():
    # Veri yolu - Örnek veri setini oluşturduğunuz dosya (lütfen önce model/data_preprocessing.py dosyasını çalıştırın)
    data_path = "model/sample_data.csv"  
    if not os.path.exists(data_path):
        print("ERROR: sample_data.csv bulunamadı! Lütfen önce 'model/data_preprocessing.py' dosyasını çalıştırın.")
        return
    
    print("Veri yükleniyor...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    print("Veri başarıyla yüklendi.")
    
    print("XGBoost modeli eğitiliyor...")
    xgb_model = train_xgboost_model(X_train, y_train)
    print("XGBoost modeli eğitildi.")
    
    print("MC Dropout modeli oluşturuluyor ve eğitiliyor...")
    mc_model = create_mc_dropout_model(X_train.shape[1])
    mc_model, history = train_mc_dropout_model(mc_model, X_train, y_train, X_test, y_test)
    print("MC Dropout modeli eğitildi.")
    
    print("Modeller kaydediliyor...")
    with open("model/model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    # HDF5 yerine yerel Keras formatında kaydediyoruz:
    mc_model.save("model/model_dropout.keras")
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Modeller başarıyla kaydedildi!")

if __name__ == "__main__":
    main()
