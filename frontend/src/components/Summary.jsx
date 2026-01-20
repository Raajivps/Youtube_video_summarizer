import React from 'react';


function Summary({ summary, audioUrl }) {
  const handleCopy = () => {
    navigator.clipboard.writeText(summary);
  };

  return (
    <div className="summary-container">
      {audioUrl && (
        <div className="audio-player-container">
          <h2>Hear the summary </h2>
          <audio controls className="audio-player" src={audioUrl}>
            Your browser does not support the audio element.
          </audio>
        </div>
      )}
      <div className="summary-text">
        <div className="summary-header">
          <h2>
            Summary 
            <button 
              onClick={() => {
                handleCopy();
                const message = document.createElement('div');
                message.textContent = 'Copied to clipboard';
                message.style.position = 'fixed';
                message.style.right = '20px';
                message.style.top = '20px';
                message.style.padding = '15px';
                message.style.backgroundColor = '#4CAF50';
                message.style.color = 'white';
                message.style.borderRadius = '4px';
                document.body.appendChild(message);
                setTimeout(() => {
                  document.body.removeChild(message);
                }, 2000);
              }} 
              className="copy-button" 
              title="Copy text" 
              style={{ 
                marginLeft: '8px',
                transition: 'background-color 0.3s ease',
                cursor: 'pointer'
              }}
            >
                📋 
                {/* copy to clipboard button */}
                
            </button>
          </h2>
        </div>
        {summary}
      </div>
    </div>
  );
}

export default Summary;