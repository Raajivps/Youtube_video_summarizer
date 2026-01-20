# YouTube Summarizer - Setup Guide

## Prerequisites
- Python 3.8+ installed
- Node.js and npm installed
- Internet connection (for downloading models and dependencies)

## Step-by-Step Setup

### 1. Backend Setup

#### Navigate to backend directory:
```bash
cd backend
```

#### Create and activate virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Install Python dependencies:
```bash
pip install -r requirements.txt
```

#### Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

#### Download NLTK data (if needed):
```bash
python -c "import nltk; nltk.download('punkt')"
```

### 2. Frontend Setup

#### Navigate to frontend directory:
```bash
cd ../frontend
```

#### Install Node.js dependencies:
```bash
npm install
```

## Running the Application

### Terminal 1 - Start Backend Server:
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

The backend will start on `http://localhost:8000`

### Terminal 2 - Start Frontend Server:
```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

## Usage

1. Open `http://localhost:3000` in your browser
2. Enter a YouTube video URL
3. Select summary type (Extractive or Abstractive)
4. Choose language
5. Click "Summarize"
6. Wait for the summary to be generated
7. Listen to the audio or copy the summary text

## Troubleshooting

### Backend Issues:
- **Module not found errors**: Make sure virtual environment is activated and dependencies are installed
- **spaCy model error**: Run `python -m spacy download en_core_web_sm`
- **CUDA errors**: The app will automatically use CPU if CUDA is not available
- **Port 8000 already in use**: Change the port in `app.py` line 427

### Frontend Issues:
- **Port 3000 already in use**: React will prompt you to use a different port
- **CORS errors**: Make sure backend is running on port 8000
- **API connection errors**: Verify backend is running and accessible

## Notes

- First run may take longer as models are downloaded
- GPU is optional but recommended for faster abstractive summarization
- Audio files are stored in `backend/static/` directory

