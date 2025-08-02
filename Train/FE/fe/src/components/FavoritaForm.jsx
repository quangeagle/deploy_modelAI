// src/components/FavoritaForm.jsx
import React, { useState } from "react";
import axios from "axios";

const initialState = {
  store_nbr: "",
  item_nbr: "",
  family: "",
  city: "",
  state: "",
  type: "",
  transactions: "",
  oil_price: "",
  day: "",
  month: "",
  dayofweek: "",
  is_weekend: "",
  is_holiday: ""
};

export default function FavoritaForm() {
  const [form, setForm] = useState(initialState);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://localhost:5000/predict_favorita", form);
      setResult(res.data);
    } catch (err) {
      setResult({ error: err.response?.data?.error || "Lỗi kết nối API" });
    }
  };

  return (
    <div>
      <h2>Dự đoán doanh thu Favorita</h2>
      <form onSubmit={handleSubmit}>
        {Object.keys(initialState).map((key) => (
          <div key={key} style={{ marginBottom: 8 }}>
            <label>
              {key}:{" "}
              <input
                type="text"
                name={key}
                value={form[key]}
                onChange={handleChange}
                required
              />
            </label>
          </div>
        ))}
        <button type="submit">Dự đoán</button>
      </form>
      {result && (
        <div style={{ marginTop: 16 }}>
          {result.error ? (
            <span style={{ color: "red" }}>{result.error}</span>
          ) : (
            <span>
              <b>Kết quả dự đoán:</b> {result.prediction}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
