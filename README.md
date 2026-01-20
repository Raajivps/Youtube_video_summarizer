# YouTube Summarizer

An AI-powered web application that generates summaries of YouTube videos with multi-language support and text-to-speech capabilities.

## Features

- 🎥 **YouTube Video Transcription** - Automatic transcript extraction from YouTube videos
- 🤖 **Dual Summarization Modes**:
  - **Extractive**: Selects and ranks key sentences from the original transcript
  - **Abstractive**: Uses T5 AI model to generate new concise summaries
- 🌍 **100+ Languages** - Translate summaries to over 100 languages
- 🔊 **Text-to-Speech** - Listen to summaries in your chosen language
- 🎨 **Modern UI** - Clean, responsive interface with dark mode support
- ⚡ **Smart Processing** - Parallel processing for long videos, intelligent chunking

## Tech Stack

### Backend
- **Python 3.8+** with Flask
- **spaCy** - NLP for extractive summarization
- **Transformers (T5)** - Abstractive summarization with AI
- **PyTorch** - Deep learning framework (CUDA support)
- **gTTS** - Google Text-to-Speech
- **deep-translator** - Multi-language translation

### Frontend
- **React 18** - Modern UI framework
- **Material-UI** - Component library
- **Axios** - HTTP client

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js and npm
- Internet connection

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd youtube-summarizer
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Frontend Setup**
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

**Terminal 1 - Backend:**
```bash
cd backend
# Activate venv first
python app.py
```
Backend runs on `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Frontend runs on `http://localhost:3000`

## Usage

1. Open `http://localhost:3000` in your browser
2. Paste a YouTube video URL
3. Select summary type (Extractive or Abstractive)
4. Choose your preferred language
5. Adjust summary length (for extractive mode)
6. Click "Summarize"
7. Listen to audio or copy the summary text

## Project Structure

```
youtube-summarizer/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── static/             # Generated audio files
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   └── components/     # UI components
│   ├── package.json        # Node dependencies
│   └── public/
├── README.md               # This file
└── SETUP.md               # Detailed setup instructions
```

## API Endpoints

- `POST /api/summarize` - Generate video summary
  ```json
  {
    "url": "youtube_url",
    "summaryType": "Extractive|Abstractive",
    "language": "English",
    "lengthPercentage": 0.3
  }
  ```
- `GET /health` - Health check endpoint
- `GET /static/<filename>` - Serve generated audio files

## Configuration

- **Backend Port**: 8000 (change in `app.py` line 427)
- **Frontend Port**: 3000 (React default)
- **CUDA Support**: Automatically detected, falls back to CPU
- **Audio Cleanup**: Old files removed after 1 hour

## Performance Optimizations

- **Transcript Caching**: LRU cache for faster repeated requests
- **Parallel Processing**: Multi-threaded summarization for long videos
- **Smart Chunking**: Automatic text splitting for efficient processing
- **Rate Limiting**: TTS request throttling to prevent API issues

## Troubleshooting

See [SETUP.md](SETUP.md) for detailed troubleshooting steps.

Common issues:
- **Port conflicts**: Change ports in configuration files
- **CUDA errors**: App automatically falls back to CPU
- **Module errors**: Ensure virtual environment is activated
- **CORS errors**: Verify backend is running on port 8000

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- YouTube Transcript API
- Hugging Face Transformers
- Google Translate & Text-to-Speech APIs
- spaCy NLP library

