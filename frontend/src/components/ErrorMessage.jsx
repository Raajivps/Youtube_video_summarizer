import React from 'react';

function ErrorMessage({ message }) {
  return (
    <div className="error-container">
      <p className="error-message">{message}</p>
    </div>
  );
}

export default ErrorMessage;