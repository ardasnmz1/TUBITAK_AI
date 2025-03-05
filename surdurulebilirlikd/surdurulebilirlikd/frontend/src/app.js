// src/App.js
import React from 'react';
import Dashboard from './dashboard.js';

function App() {
  return (
    <div className="App">
      <header style={{ textAlign: 'center', margin: '20px' }}>
        <h1>Sürdürülebilirlik Öneri Sistemi</h1>
      </header>
      <Dashboard />
    </div>
  );
}

export default App;
