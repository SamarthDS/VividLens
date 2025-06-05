import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import ModelDetails from './pages/ModelDetails';
import Layout from './components/Layout';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="/models" element={<ModelDetails />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;