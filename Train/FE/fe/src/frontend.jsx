import React, { useState } from 'react';
import WalmartForm from './components/WalmartForm';
import FavoritaForm from './components/FavoritaForm';


export default function App() {
  const [selectedModel, setSelectedModel] = useState('');

  return (
    <div className="min-h-screen p-6 bg-gray-100">
      <div className="max-w-xl mx-auto bg-white p-6 rounded-xl shadow-xl">
        <h1 className="text-2xl font-bold text-center mb-4">Chọn hệ thống dự đoán</h1>

        <div className="flex justify-center space-x-4 mb-6">
        <button onClick={() => setSelectedModel('walmart')}>
  Walmart
</button>

          <button onClick={() => setSelectedModel('favorita')}>
  Favorita
</button>

        </div>

        {selectedModel === 'walmart' && <WalmartForm />}
        {selectedModel === 'favorita' && <FavoritaForm />}
      </div>
    </div>
  );
}
