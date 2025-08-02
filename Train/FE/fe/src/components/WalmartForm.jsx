// components/WalmartForm.jsx
import React, { useState } from 'react';
import axios from 'axios';

const WalmartForm = () => {
  const [formData, setFormData] = useState({
    model: 'XGBoost',
    Store: '',
    Holiday_Flag: '',
    Temperature: '',
    Fuel_Price: '',
    CPI: '',
    Unemployment: ''
  });

  const [result, setResult] = useState(null);
  const models = ['XGBoost', 'LightGBM', 'CatBoost'];

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:5000/predict', formData);
      setResult(res.data);
    } catch (err) {
      setResult({ error: err.response?.data?.error || 'Lỗi không xác định' });
    }
  };

  return (
    <div className="form-container">
      <h2 className="text-xl font-bold mb-4">Dự đoán doanh thu Walmart</h2>
      <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
        <select name="model" value={formData.model} onChange={handleChange} className="col-span-2 p-2 border">
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>

        {['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'].map((field) => (
          <input
            key={field}
            type="number"
            name={field}
            value={formData[field]}
            onChange={handleChange}
            placeholder={field}
            className="p-2 border"
            required
          />
        ))}

        <button type="submit" className="col-span-2 bg-blue-500 text-white p-2 rounded hover:bg-blue-600">
          Dự đoán
        </button>
      </form>

      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          {result.error ? (
            <p className="text-red-600">{result.error}</p>
          ) : (
            <p>Kết quả dự đoán ({result.model}): <strong>{result.prediction}</strong></p>
          )}
        </div>
      )}
    </div>
  );
};

export default WalmartForm;
