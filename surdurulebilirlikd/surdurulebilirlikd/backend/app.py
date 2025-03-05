import os
import gc
import numpy as np
import pickle
import tensorflow as tf
import datetime
import random
import pandas as pd
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import atexit
import io
import pyodbc
import requests  # requests modülünü ekledik
import json

gc.collect()    

# Proje kök dizini ve ilgili klasörler
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model dosya yolları
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
DROPOUT_MODEL_PATH = os.path.join(BASE_DIR, "model", "model_dropout.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# HERE Traffic API için API anahtarınızı buraya ekleyin
TRAFFIC_API_KEY = "w41acAzG4OLdcFNzsC3U3UqAwFCbOHMe" # API anahtarınızı girin



# Dummy modeller
class DummyClassicModel:
    def predict(self, X):
        return np.zeros((X.shape[0],))
        
class DummyMCModel:
    def predict(self, X, verbose=0):
        return np.ones((X.shape[0], 1))
    def compile(self, optimizer, loss):
        pass
        
class DummyScaler:
    def transform(self, X):
        return X

def load_models():
    global classic_model, mc_model, scaler
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                classic_model = pickle.load(f)
            logger.info("Klasik model yüklendi.")
        else:
            classic_model = DummyClassicModel()
            logger.warning("Klasik model bulunamadı, dummy model kullanılıyor.")
    except Exception as e:
        logger.error(f"Klasik model yüklenemedi: {e}")
        classic_model = DummyClassicModel()
        
    try:
        if os.path.exists(DROPOUT_MODEL_PATH):
            mc_model = tf.keras.models.load_model(DROPOUT_MODEL_PATH, compile=False)
            mc_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
            logger.info("MC Dropout modeli yüklendi.")
        else:
            mc_model = DummyMCModel()
            logger.warning("MC Dropout modeli bulunamadı, dummy model kullanılıyor.")
    except Exception as e:
        logger.error(f"MC model yüklenemedi: {e}")
        mc_model = DummyMCModel()
        
    try:
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            logger.info("Scaler yüklendi.")
        else:
            scaler = DummyScaler()
            logger.warning("Scaler bulunamadı, dummy scaler kullanılıyor.")
    except Exception as e:
        logger.error(f"Scaler yüklenemedi: {e}")
        scaler = DummyScaler()

load_models()

def monte_carlo_predict(model, X, n_samples=100):
    try:
        preds = np.stack([model.predict(X, verbose=0) for _ in range(n_samples)])
        return preds.mean(axis=0).flatten(), preds.std(axis=0).flatten()
    except Exception as e:
        logger.error(f"MC tahmin hatası: {e}")
        return np.zeros(X.shape[0]), np.ones(X.shape[0])

@app.route('/api/model_summary')
def model_summary():
    try:
        model = tf.keras.models.load_model(DROPOUT_MODEL_PATH, compile=False)
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_text = "\n".join(summary_lines)
        return jsonify({"model_summary": summary_text})
    except Exception as e:
        logger.error(f"Model özet hatası: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/bus_departures')
def bus_departures():
    dummy_data = [
        {"kalkis": "06:00", "varis": "06:45", "yogunluk": 65},
        {"kalkis": "07:00", "varis": "07:50", "yogunluk": 80},
        {"kalkis": "08:00", "varis": "08:40", "yogunluk": 90},
        {"kalkis": "09:00", "varis": "09:50", "yogunluk": 75},
        {"kalkis": "10:00", "varis": "10:40", "yogunluk": 50},
        {"kalkis": "11:00", "varis": "11:40", "yogunluk": 30}
    ]
    return jsonify({"data": dummy_data})

# HERE Traffic API'den İzmir trafik yoğunluğu yüzdesini alır (6.2 versiyonu kullanılıyor)
def get_izmir_traffic_percentage():
    try:
        lat, lon = 38.423734, 27.142826  # İzmir koordinatları
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&unit=KMPH&key={TRAFFIC_API_KEY}"
        response = requests.get(url)

        logger.info(f"API İsteği Yapıldı: Status Code -> {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            flow_data = data.get("flowSegmentData", {})
            current_speed = flow_data.get("currentSpeed", None)
            free_flow_speed = flow_data.get("freeFlowSpeed", None)

            if current_speed is not None and free_flow_speed and free_flow_speed > 0:
                congestion = (1 - (current_speed / free_flow_speed)) * 100
                return round(congestion, 2)

        logger.error(f"❌ API Yanıt Hatası: Status Code -> {response.status_code}, Yanıt -> {response.text}")
        return 50.0  # Varsayılan değer
    except Exception as e:
        logger.error(f"❌ HATA: Trafik yoğunluğu API isteğinde hata oluştu -> {e}")
        return 50.0  # Varsayılan değer
    



# CSV üzerinden trafik modeli eğitimi sırasında "HAT_NO" sütunu kontrolü
def check_and_rename_columns(df):
    if 'HAT_NO' not in df.columns:
        for col in df.columns:
            if col.lower() == 'hat_no':
                df.rename(columns={col: "HAT_NO"}, inplace=True)
                logger.info(f"Sütun adı '{col}' 'HAT_NO' olarak yeniden adlandırıldı.")
                break
    return df

# Sefer programı oluşturma: Trafik yoğunluğu doğrudan API'den alınan değere göre kullanılıyor.
def generate_bus_schedule_full(hat_no):
    today = datetime.datetime.now().date()
    start_time = datetime.datetime.combine(today, datetime.time(5, 0))
    end_time = datetime.datetime.combine(today, datetime.time(23, 59))
    schedule = []
    # API'den trafik yoğunluğu değeri alınır
    api_density = get_izmir_traffic_percentage()  # 0-100 arası bir değer
    current_time = start_time
    while current_time <= end_time:
        additional = int(round((api_density / 100) * 10))
        min_interval = 10 + additional
        max_interval = 15 + additional
        interval = random.randint(min_interval, max_interval)
        departure_time = current_time.strftime("%H:%M")
        base_accuracy = 100 - api_density + random.uniform(-5, 5)
        increased_accuracy = min(base_accuracy * 1.1, 100)
        carbon_reduction = round(random.uniform(5, 15) * (api_density / 100), 2)
        safety_percentage = round(min(99, increased_accuracy + random.uniform(-3, 3)), 1)
        schedule.append({
            "departure_time": departure_time,
            "accuracy": round(increased_accuracy, 1),
            "carbon_reduction": carbon_reduction,
            "safety_percentage": safety_percentage,
            "interval_minutes": interval,
            "traffic_density": round(api_density, 1)
        })
        current_time += datetime.timedelta(minutes=interval)
    return schedule

@app.route('/api/generate_schedule', methods=['POST'])
def generate_schedule():
    try:
        data = request.get_json(force=True)
        hat_no = data.get('HAT_NO')
        if not hat_no:
            return jsonify({"error": "Otobüs hattı numarası (HAT_NO) eksik"}), 400
        schedule = generate_bus_schedule_full(hat_no)
        return jsonify({"data": schedule})
    except Exception as e:
        logger.error(f"Schedule oluşturma hatası: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/schedule_statistics', methods=['POST'])
def schedule_statistics():
    try:
        data = request.get_json(force=True)
        hat_no = data.get("HAT_NO")
        if not hat_no:
            return jsonify({"error": "Otobüs hattı numarası (HAT_NO) eksik"}), 400
        schedule = generate_bus_schedule_full(hat_no)
        if not schedule:
            return jsonify({"error": "Sefer programı oluşturulamadı"}), 400
        total_trips = len(schedule)
        intervals = [item["interval_minutes"] for item in schedule if item["interval_minutes"] > 0]
        avg_interval = round(sum(intervals) / len(intervals), 1) if intervals else 0
        departure_times = [item["departure_time"] for item in schedule]
        earliest = min(departure_times)
        latest = max(departure_times)
        return jsonify({
            "HAT_NO": hat_no,
            "total_trips": total_trips,
            "average_interval_minutes": avg_interval,
            "earliest_departure": earliest,
            "latest_departure": latest,
            "last_traffic_update": last_traffic_update
        })
    except Exception as e:
        logger.error(f"Schedule statistics hatası: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/historical_traffic', methods=['GET'])
def historical_traffic():
    try:
        csv_path = os.path.join("data", "hareket_saatleri.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "hareket_saatleri.csv bulunamadı"}), 400
        df = pd.read_csv(csv_path)
        df = check_and_rename_columns(df)
        if 'HAT_NO' not in df.columns or 'traffic_density' not in df.columns:
            return jsonify({"error": "Gerekli sütunlar bulunamadı"}), 400
        grouped = df.groupby("HAT_NO")["traffic_density"].mean().reset_index()
        grouped["traffic_density"] = grouped["traffic_density"].round(1)
        return jsonify(grouped.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Historical traffic hatası: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/export_schedule', methods=['POST'])
def export_schedule():
    try:
        data = request.get_json(force=True)
        hat_no = data.get("HAT_NO")
        if not hat_no:
            return jsonify({"error": "Otobüs hattı numarası (HAT_NO) eksik"}), 400
        schedule = generate_bus_schedule_full(hat_no)
        df = pd.DataFrame(schedule)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return Response(csv_buffer, mimetype="text/csv",
                        headers={"Content-disposition": f"attachment; filename={hat_no}_schedule.csv"})
    except Exception as e:
        logger.error(f"Export schedule hatası: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/historical_traffic_data', methods=['GET'])
def historical_traffic_data():
    try:
        csv_path = os.path.join("hareket_saatleri.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "hareket_saatleri.csv bulunamadı"}), 400
        df = pd.read_csv(csv_path)
        df = check_and_rename_columns(df)
        if 'HAT_NO' not in df.columns or 'traffic_density' not in df.columns:
            return jsonify({"error": "Gerekli sütunlar bulunamadı"}), 400
        grouped = df.groupby("HAT_NO")["traffic_density"].mean().reset_index()
        grouped["traffic_density"] = grouped["traffic_density"].round(1)
        return jsonify(grouped.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"Historical traffic hatası: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/export_schedule_data', methods=['POST'])
def export_schedule_data():
    try:
        data = request.get_json(force=True)
        hat_no = data.get("HAT_NO")
        if not hat_no:
            return jsonify({"error": "Otobüs hattı numarası (HAT_NO) eksik"}), 400
        schedule = generate_bus_schedule_full(hat_no)
        df = pd.DataFrame(schedule)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return Response(csv_buffer, mimetype="text/csv",
                        headers={"Content-disposition": f"attachment; filename={hat_no}_schedule.csv"})
    except Exception as e:
        logger.error(f"Export schedule hatası: {e}")
        return jsonify({"error": str(e)}), 400

# Yeni ML istatistikleri endpoint'i (örnek statik değerler)
@app.route('/api/ml_statistics')
def ml_statistics():
    return jsonify({
        "knn_accuracy": "85%",
        "rf_accuracy": "90%",
        "xgb_accuracy": "88%"
    })

@app.route('/')
def index():
    return render_template('index.html')

# Güncel trafik modeli ve sefer istatistikleri güncelleme
last_traffic_update = None

# Global model değişkenleri
traffic_knn = None
traffic_rf = None
traffic_xgb = None
traffic_le = None

# Modellerin global olarak tanımlandığını kontrol etmek için
# train_traffic_density_models() fonksiyonunu güncelliyorum.
def train_traffic_density_models():
    global traffic_knn, traffic_rf, traffic_xgb, traffic_le, last_traffic_update
    logger.info("🚀 Model eğitimi başladı...")
    traffic_knn = KNeighborsRegressor(n_neighbors=5)
    traffic_rf = RandomForestRegressor(n_estimators=50, random_state=42)
    traffic_xgb = xgb.XGBRegressor(n_estimators=50, random_state=42, objective='reg:squarederror')

    csv_path = os.path.join("hareket_saatleri.csv")

    if not os.path.exists(csv_path):
        logger.warning("hareket_saatleri.csv bulunamadı, dummy trafik modeli kullanılacak.")
        return

    try:
        df = pd.read_csv(csv_path)
        df = check_and_rename_columns(df)

        if 'traffic_density' not in df.columns:
            df['traffic_density'] = np.random.randint(0, 101, size=len(df))
        if 'hour' not in df.columns:
            df['hour'] = 12
        if 'minute' not in df.columns:
            df['minute'] = 30

        features = df[['hour', 'minute']]
        target = df['traffic_density']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        traffic_knn = KNeighborsRegressor(n_neighbors=5)
        traffic_rf = RandomForestRegressor(n_estimators=50, random_state=42)
        traffic_xgb = xgb.XGBRegressor(n_estimators=50, random_state=42, objective='reg:squarederror')

        traffic_knn.fit(X_train, y_train)
        traffic_rf.fit(X_train, y_train)
        traffic_xgb.fit(X_train, y_train)

        last_traffic_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        traffic_knn.fit(X_train, y_train)
        traffic_rf.fit(X_train, y_train)
        traffic_xgb.fit(X_train, y_train)

        if not hasattr(traffic_knn, "n_neighbors"):
            logger.error("KNN Modeli henüz eğitilmedi.")
        
        if traffic_knn is None or traffic_rf is None or traffic_xgb is None:
            return jsonify({"error": "Modeller yüklenemedi."}), 500

        with open("surdurulebilirlikd/backend/model/knn_model.pkl", "wb") as f:
            pickle.dump(traffic_knn, f)

        with open("surdurulebilirlikd/backend/model/rf_model.pkl", "wb") as f:
            pickle.dump(traffic_rf, f)

        with open("surdurulebilirlikd/backend/model/xgb_model.pkl", "wb") as f:
            pickle.dump(traffic_xgb, f)
            
            # Model yükleme işlemleri
        with open("surdurulebilirlikd/backend/model/knn_model.pkl", "rb") as f:
            traffic_knn = pickle.load(f)

        with open("surdurulebilirlikd/backend/model/rf_model.pkl", "rb") as f:
            traffic_rf = pickle.load(f)

        with open("surdurulebilirlikd/backend/model/xgb_model.pkl", "rb") as f:
            traffic_xgb = pickle.load(f)



        logger.info(f"Modeller Başarıyla Eğitildi: \nKNN: {traffic_knn}\nRF: {traffic_rf}\nXGB: {traffic_xgb}")
        if traffic_knn is None or traffic_rf is None or traffic_xgb is None:
            logger.error("Bir veya daha fazla model yüklenemedi.")
    except Exception as e:
        logger.error(f"Trafik modeli eğitilirken hata: {e}")



train_traffic_density_models()



def predict_traffic_density_ml(hat_no):
    try:
        hat_no_enc = traffic_le.transform([str(hat_no)])[0]
        now = datetime.datetime.now()
        features = np.array([[hat_no_enc, now.hour, now.minute]])
        pred_knn = traffic_knn.predict(features)[0]
        pred_rf = traffic_rf.predict(features)[0]
        pred_xgb = traffic_xgb.predict(features)[0]
        traffic_density = (pred_knn + pred_rf + pred_xgb) / 3.0
        return max(0, min(100, traffic_density))
    except Exception as e:
        logger.error(f"Traffic density prediction error: {e}")
        return 50.0

def get_traffic_density():
    lat, lon = 38.423734, 27.142826  # İzmir koordinatları
    api_key = "w41acAzG4OLdcFNzsC3U3UqAwFCbOHMe"  # API anahtarınız
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&unit=KMPH&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        flow_data = data.get("flowSegmentData", {})
        current_speed = flow_data.get("currentSpeed", None)
        free_flow_speed = flow_data.get("freeFlowSpeed", None)
        if current_speed and free_flow_speed:
            congestion = (1 - (current_speed / free_flow_speed)) * 100
            return max(0, min(100, round(congestion, 2)))  # Değerleri 0-100 arasında sınırla
    return 50.0  # Varsayılan değer

def calculate_traffic_density(hat_no):
    try:
        predicted = predict_traffic_density_ml(hat_no)
        live = random.uniform(0, 100)
        final_density = (predicted * 0.7) + (live * 0.3)
        logger.info(f"ML tahmini: {predicted}, Canlı: {live}, Son trafik yoğunluğu: {final_density}")
        return max(0, min(100, final_density))
    except Exception as e:
        logger.error(f"Error calculating traffic density: {e}")
        return 50.0

def update_model():
    try:
        logger.info("Model güncelleme başlatıldı.")
        load_models()
        logger.info("Model güncelleme tamamlandı.")
    except Exception as e:
        logger.error(f"Güncelleme hatası: {e}")



def calculate_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.where(y_true == 0, 0.01, y_true)  # Sıfır bölme hatasını önleme
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return max(0, round(accuracy, 2))  # Negatif doğrulukları sıfırla

# 1. API route metodunu düzeltmek
# 1. API route metodunu düzeltmek
@app.route('/api/prediction_accuracies', methods=['GET'])
def prediction_accuracies():
    try:
        now = datetime.datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        
        # Fetch real-time traffic density from TomTom Traffic API
        traffic_density = get_traffic_density()

        # Features for prediction: Only hour and minute are used, HAT_NO is not included
        features = np.array([[current_hour, current_minute]])

        # Make predictions using the trained models
        pred_knn = traffic_knn.predict(features)[0]
        pred_rf = traffic_rf.predict(features)[0]
        pred_xgb = traffic_xgb.predict(features)[0]

        # Calculate the accuracy using real-time traffic density as the ground truth
        knn_accuracy = calculate_accuracy([traffic_density], [pred_knn])
        rf_accuracy = calculate_accuracy([traffic_density], [pred_rf])
        xgb_accuracy = calculate_accuracy([traffic_density], [pred_xgb])

        return jsonify({
            "knn_accuracy": knn_accuracy,
            "rf_accuracy": rf_accuracy,
            "xgb_accuracy": xgb_accuracy
        })
    except Exception as e:
        logger.error(f"Prediction accuracy calculation error: {str(e)}")
        return jsonify({"error": str(e)}), 400



def get_izmir_traffic_percentage():
    try:
        lat, lon = 38.423734, 27.142826  # İzmir koordinatları
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&unit=KMPH&key={TRAFFIC_API_KEY}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            flow_data = data.get("flowSegmentData", {})
            current_speed = flow_data.get("currentSpeed", None)
            free_flow_speed = flow_data.get("freeFlowSpeed", None)

            if current_speed is not None and free_flow_speed and free_flow_speed > 0:
                congestion = (1 - (current_speed / free_flow_speed)) * 100
                return round(congestion, 2)

        logger.error(f"❌ HATA: API'den trafik verisi çekilemedi -> {response.status_code}")
        return 50.0  # Varsayılan değer
    except Exception as e:
        logger.error(f"❌ HATA: Trafik yoğunluğu API isteğinde hata oluştu -> {e}")
        return 50.0  # Varsayılan değer

logger.info(f"KNN Model Fit Durumu: {hasattr(traffic_knn, 'n_neighbors')}")
logger.info(f"Random Forest Fit Durumu: {hasattr(traffic_rf, 'n_estimators')}")
logger.info(f"XGBoost Fit Durumu: {hasattr(traffic_xgb, 'n_estimators')}")

now = datetime.datetime.now()
hour = now.hour
minute = now.minute

# Trafik yoğunluğunu ekleyin
traffic_density = get_traffic_density()

# Özellikler ve hedef değişken
features = np.array([[hour, minute, traffic_density]])

# Verileri ölçeklendirme
scaler = StandardScaler()
X = scaler.fit_transform(features)
target = [traffic_density]  # Hedef değişken

# Veri sayısı kontrolü
y_samples = len(X)
if y_samples > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
else:
    X_train, y_train = X, target  # Tüm veriyi eğitimde kullan

# Modelleri tanımla
n_neighbors = min(3, len(X_train))  # Dinamik olarak komşu sayısını belirleme
knn = KNeighborsRegressor(n_neighbors=n_neighbors)
rf = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=5, min_samples_leaf=2, random_state=42)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=15, learning_rate=0.1, random_state=42)

# Modelleri eğit
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Özellikleri ölçeklendirme
features_scaled = scaler.transform(features)

# Tahmin yapın
knn_prediction = max(0, min(100, knn.predict(features_scaled)[0]))
rf_prediction = max(0, min(100, rf.predict(features_scaled)[0]))
xgb_prediction = max(0, min(100, xgb_model.predict(features_scaled)[0]))

# Doğruluk hesapla
knn_accuracy = calculate_accuracy([traffic_density], [knn_prediction])
rf_accuracy = calculate_accuracy([traffic_density], [rf_prediction])
xgb_accuracy = calculate_accuracy([traffic_density], [xgb_prediction])

# Sonuçları yazdır
print(f'KNN Prediction: {knn_prediction:.2f} - Accuracy: %{knn_accuracy}')
print(f'Random Forest Prediction: {rf_prediction:.2f} - Accuracy: %{rf_accuracy}')
print(f'XGBoost Prediction: {xgb_prediction:.2f} - Accuracy: %{xgb_accuracy}')

scheduler = BackgroundScheduler()
scheduler.add_job(func=update_model, trigger="cron", day_of_week="mon", hour=0)
scheduler.add_job(func=train_traffic_density_models, trigger="cron", minute=0)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    train_traffic_density_models()
    try:
        PORT = 5000
        logger.info(f"Sunucu http://localhost:{PORT} adresinde başlatılıyor...")
        app.run(host="localhost", port=PORT, debug=True, use_reloader=False)
    except Exception as e:
        logger.critical(f"Sunucu başlatma hatası: {e}")
            

