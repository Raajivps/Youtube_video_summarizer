import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import Summary from './components/Summary';
import VideoPlayer from './components/VideoPlayer';
import ErrorMessage from './components/ErrorMessage';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [summaryType, setSummaryType] = useState('Extractive');
  const [language, setLanguage] = useState('English');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
    }
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  const handleSummarize = async (lengthPercentage) => {
    if (!url.trim()) {
      setError('Please enter a valid URL');
      return;
    }

    setLoading(true);
    setError(null);
    setSummary('');
    setAudioUrl(null);

    try {
      const response = await axios.post('http://localhost:8000/api/summarize', {
        url,
        summaryType,
        language,
        lengthPercentage: lengthPercentage
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.data.success) {
        setSummary(response.data.summary);
        if (response.data.audio_url) {
          setAudioUrl(`http://localhost:8000${response.data.audio_url}`);
        }
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'An unexpected error occurred. Please try again later.';
      setError(errorMessage);
      console.error('Summarization error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      
      <Sidebar 
        url={url}
        setUrl={setUrl}
        summaryType={summaryType}
        setSummaryType={setSummaryType}
        language={language}
        setLanguage={setLanguage}
        onSummarize={handleSummarize}
        loading={loading}
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
      />
      <main className="main-content">
        {error && <ErrorMessage message={error} />}
        {url && <VideoPlayer url={url} />}
        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Generating summary...</p>
          </div>
        )}
        {!loading && summary && <Summary summary={summary} audioUrl={audioUrl} />}
      </main>
    </div>
  );
}

export default App;