import React, { useState } from 'react';

function Sidebar({ 
  url, 
  setUrl, 
  summaryType, 
  setSummaryType,
  language,
  setLanguage,
  onSummarize,
  loading,
  isDarkMode,
  toggleTheme
}) {
  const languages_dict = {
    'en': 'English',
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'am': 'Amharic',
    'ar': 'Arabic',
    'hy': 'Armenian',
    'az': 'Azerbaijani',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bn': 'Bengali',
    'bs': 'Bosnian',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'ceb': 'Cebuano',
    'ny': 'Chichewa',
    'zh-cn': 'Chinese (simplified)',
    'zh-tw': 'Chinese (traditional)',
    'co': 'Corsican',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'eo': 'Esperanto',
    'et': 'Estonian',
    'tl': 'Filipino',
    'fi': 'Finnish',
    'fr': 'French',
    'fy': 'Frisian',
    'gl': 'Galician',
    'ka': 'Georgian',
    'de': 'German',
    'el': 'Greek',
    'gu': 'Gujarati',
    'ht': 'Haitian creole',
    'ha': 'Hausa',
    'haw': 'Hawaiian',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hmn': 'Hmong',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'ig': 'Igbo',
    'id': 'Indonesian',
    'ga': 'Irish',
    'it': 'Italian',
    'ja': 'Japanese',
    'jw': 'Javanese',
    'kn': 'Kannada',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'ko': 'Korean',
    'ku': 'Kurdish (kurmanji)',
    'ky': 'Kyrgyz',
    'lo': 'Lao',
    'la': 'Latin',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'lb': 'Luxembourgish',
    'mk': 'Macedonian',
    'mg': 'Malagasy',
    'ms': 'Malay',
    'ml': 'Malayalam',
    'mt': 'Maltese',
    'mi': 'Maori',
    'mr': 'Marathi',
    'mn': 'Mongolian',
    'my': 'Myanmar (burmese)',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'or': 'Odia',
    'ps': 'Pashto',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'pa': 'Punjabi',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sm': 'Samoan',
    'gd': 'Scots gaelic',
    'sr': 'Serbian',
    'st': 'Sesotho',
    'sn': 'Shona',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'so': 'Somali',
    'es': 'Spanish',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'tg': 'Tajik',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'ug': 'Uyghur',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'cy': 'Welsh',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'zu': 'Zulu'
  };

  const [lengthPercentage, setLengthPercentage] = useState(30); // default 30%
  //const [error, setError] = useState(null); // Add error state if needed

  const handleSummarize = () => {
    // Pass the lengthPercentage to the parent's onSummarize
    onSummarize(lengthPercentage / 100); // Convert to decimal
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>YouTube Video Summarizer</h2>
        <button 
          onClick={toggleTheme} 
          className="theme-button"
          title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDarkMode ? '☀️' : '🌙'}
        </button>
      </div>
      
      <img src="/app_logo.gif" alt="App Logo" className="app-logo" style={{ width: '100%', maxWidth: '400px', height: 'auto', marginBottom: '20px', borderRadius: '12px', border: '2px solid var(--border-color)' }} />
      
      <div className="input-group">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter YouTube video URL"
          className="url-input"
        />

        <select 
          value={summaryType}
          onChange={(e) => setSummaryType(e.target.value)}
          className="select-input"
        >
          <option value="Extractive">Extractive (Spacy Algorithm)</option>
          <option value="Abstractive">Abstractive (T5 Algorithm)</option>
        </select>

        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="select-input"
        >
          {Object.values(languages_dict).map((lang) => (
            <option key={lang} value={lang}>{lang}</option>
          ))}
        </select>

        <button 
          onClick={handleSummarize} 
          disabled={loading || !url}
          className="summarize-btn"
        >
          {loading ? 'Summarizing...' : 'Summarize'}
        </button>
      </div>

      {summaryType === "Extractive" && (
        <div className="form-group">
          <label htmlFor="length-percentage">Summary Length (%)</label>
          <input
            type="range"
            id="length-percentage"
            min="10"
            max="50"
            value={lengthPercentage}
            onChange={(e) => setLengthPercentage(Number(e.target.value))}
            className="range-input"
          />
          <span>{lengthPercentage}%</span>
        </div>
      )}
    </div>
  );
}

export default Sidebar;