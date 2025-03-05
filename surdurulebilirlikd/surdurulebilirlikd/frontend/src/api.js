// src/api.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000' // Flask API URL'sini kendi ortamınıza göre ayarlayın
});

export default api;
