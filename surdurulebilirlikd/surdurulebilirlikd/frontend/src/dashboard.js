// src/Dashboard.js
import React, { useState } from 'react';
import api from './api';
import { Bar } from 'react-chartjs-2';
import 'chart.js/auto';

function Dashboard() {
  const [inputData, setInputData] = useState('[[0.1, 0.2, 0.3, 0.4]]');
  const [result, setResult] = useState(null);
  const [chartData, setChartData] = useState(null);

  const sendData = async () => {
    try {
      const inputs = JSON.parse(inputData);
      const response = await api.post('/predict', { inputs });
      setResult(response.data);
      // Grafik verilerini oluşturmak için
      const labels = response.data.mc_prediction_mean.map((_, i) => `Örnek ${i + 1}`);
      setChartData({
        labels: labels,
        datasets: [{
          label: 'Monte Carlo Tahmin Ortalaması',
          data: response.data.mc_prediction_mean,
          backgroundColor: 'rgba(75, 192, 192, 0.6)'
        }]
      });
    } catch (error) {
      setResult({ error: error.message });
    }
  };

  return (
    <div style={{ padding: '0 20px' }}>
      <textarea
        rows="6"
        cols="80"
        value={inputData}
        onChange={(e) => setInputData(e.target.value)}
      ></textarea>
      <br />
      <button onClick={sendData} style={{ marginTop: '10px', padding: '10px 20px' }}>
        Tahmin Al
      </button>
      {result && result.error && <p style={{ color: 'red' }}>{result.error}</p>}
      {result && !result.error && (
        <div>
          <h3>Tahmin Sonuçları</h3>
          <ul>
            {result.classic_prediction.map((pred, i) => (
              <li key={i}>Örnek {i + 1}: XGBoost Tahmini = {pred.toFixed(2)}</li>
            ))}
          </ul>
          <h3>Monte Carlo Dropout Tahminleri</h3>
          <ul>
            {result.mc_prediction_mean.map((mean, i) => (
              <li key={i}>
                Örnek {i + 1}: Tahmin = {mean.toFixed(2)} ± {result.mc_prediction_std[i].toFixed(2)}<br />
                %95 Güven Aralığı: [{result.lower_bound[i].toFixed(2)}, {result.upper_bound[i].toFixed(2)}]
              </li>
            ))}
          </ul>
        </div>
      )}
      {chartData && (
        <div style={{ maxWidth: '800px', marginTop: '20px' }}>
          <Bar data={chartData} options={{ responsive: true, scales: { y: { beginAtZero: true } } }} />
        </div>
      )}
    </div>
  );
}

export default Dashboard;
