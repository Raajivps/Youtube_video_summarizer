from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from gtts import gTTS
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from functools import lru_cache
import time
import nltk
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from pydub import AudioSegment
from threading import Lock


app = Flask(__name__)
CORS(app)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Language mapping for translation and TTS
LANGUAGE_CODES = {
    'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy',
    'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs',
    'Bulgarian': 'bg', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chinese (simplified)': 'zh-CN',
    'Chinese (traditional)': 'zh-TW', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs',
    'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et',
    'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy', 'Galician': 'gl', 'Georgian': 'ka',
    'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian creole': 'ht', 'Hausa': 'ha',
    'Hawaiian': 'haw', 'Hebrew': 'he', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu',
    'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it',
    'Japanese': 'ja', 'Javanese': 'jv', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km',
    'Korean': 'ko', 'Kurdish (kurmanji)': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la',
    'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk',
    'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi',
    'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar (burmese)': 'my', 'Nepali': 'ne',
    'Norwegian': 'no', 'Odia': 'or', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl',
    'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm',
    'Scots gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 'Shona': 'sn', 'Sindhi': 'sd',
    'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es',
    'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta',
    'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur',
    'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh',
    'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu',
    'Chinese': 'zh-CN',
    'Chinese (Simplified)': 'zh-CN', 
    'Chinese (Traditional)': 'zh-TW'
}

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
nlp = spacy.load("en_core_web_sm")
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# Cache for transcripts
@lru_cache(maxsize=100)
def get_cached_transcript(video_id):
    return YouTubeTranscriptApi.get_transcript(video_id)

def translate_text(text, target_lang):
    """Translate text with parallel processing for chunks"""
    try:
        chunk_size = 4000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        translator = GoogleTranslator(source='en', target=target_lang)
        
        # Use ThreadPoolExecutor for parallel translation
        with ThreadPoolExecutor(max_workers=min(len(chunks), 5)) as executor:
            translate_chunk = partial(translator.translate)
            translated_chunks = list(executor.map(translate_chunk, chunks))
                
        return ' '.join(filter(None, translated_chunks))
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def preprocess_transcript(text):
    """Clean and preprocess the transcript text"""
    # Remove special markers and unnecessary whitespace
    text = ' '.join(
        line for line in text.split()
        if not any(marker in line.lower() for marker in ['[music]', '[applause]', '[laughter]', '[silence]'])
    )
    
    # Remove multiple spaces and normalize text
    text = ' '.join(text.split())
    return text

def extractive_summarize(text, length_percentage=0.3):
    """Optimized extractive summarization for long texts"""
    # Preprocess text
    text = preprocess_transcript(text)
    words = text.split()
    
    # Ensure minimum and maximum summary lengths
    min_sentences = 3
    max_sentences = 15  # Add a maximum cap
    
    if len(words) > 5000:
        chunks = []
        for i in range(0, len(words), 5000):
            chunk = ' '.join(words[i:i + 5000])
            chunks.append(chunk)
        
        # Process each chunk in parallel
        with ThreadPoolExecutor(max_workers=min(len(chunks), 5)) as executor:
            chunk_summaries = []
            for chunk in chunks:
                future = executor.submit(process_chunk_extractive, chunk, length_percentage)
                chunk_summaries.append(future)
            
            # Combine results
            all_sentences = []
            for future in chunk_summaries:
                all_sentences.extend(future.result())
            
            # Sort sentences by score and get top ones
            sorted_sentences = sorted(all_sentences, key=lambda x: x[1], reverse=True)
            # Calculate number of sentences based on percentage but with limits
            n_sentences = min(
                max(
                    int(len(sorted_sentences) * length_percentage),
                    min_sentences
                ),
                max_sentences
            )
            selected_sentences = sorted_sentences[:n_sentences]
            
            # Extract just the sentences (without scores) and join them
            final_summary = ' '.join(sent[0] for sent in selected_sentences)
            return final_summary
    
    # For shorter texts, process directly
    sentences_with_scores = process_chunk_extractive(text, length_percentage)
    sorted_sentences = sorted(sentences_with_scores, key=lambda x: x[1], reverse=True)
    n_sentences = min(
        max(
            int(len(sorted_sentences) * length_percentage),
            min_sentences
        ),
        max_sentences
    )
    selected_sentences = sorted_sentences[:n_sentences]
    return ' '.join(sent[0] for sent in selected_sentences)

def process_chunk_extractive(text, length_percentage):
    """Process a single chunk for extractive summarization"""
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_scores = []
    
    for i, sent in enumerate(sentences):
        # Calculate score
        score = 0
        entities = len([ent for ent in sent.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']])
        noun_phrases = len([chunk for chunk in sent.noun_chunks])
        
        score = entities * 2 + noun_phrases
        
        # Position scoring
        if i < len(sentences) * 0.2:
            score *= 1.3
        elif i > len(sentences) * 0.8:
            score *= 1.2
        
        # Store sentence text and its score
        sentence_scores.append((sent.text.strip(), score))
    
    # Return all sentences with their scores
    return sentence_scores

def abstractive_summarize(text, max_length=150, min_length=50):
    """Optimized abstractive summarization for long texts"""
    text = preprocess_transcript(text)
    words = text.split()
    
    # For very long texts, use a different approach
    if len(words) > 10000:
        # Split into major sections
        section_size = 5000
        sections = []
        
        for i in range(0, len(words), section_size):
            section = ' '.join(words[i:i + section_size])
            sections.append(section)
        
        # Process sections in parallel
        with ThreadPoolExecutor(max_workers=min(len(sections), 3)) as executor:
            futures = []
            for section in sections:
                future = executor.submit(
                    process_chunk_abstractive,
                    section,
                    max(max_length // 2, 100),  # Shorter summaries for sections
                    min(min_length // 2, 30)
                )
                futures.append(future)
            
            # Combine section summaries
            section_summaries = []
            for future in futures:
                section_summaries.append(future.result())
        
        # Final combination summary
        combined_text = " ".join(section_summaries)
        return process_chunk_abstractive(combined_text, max_length, min_length)
    
    return process_chunk_abstractive(text, max_length, min_length)

def process_chunk_abstractive(text, max_length, min_length):
    """Process a single chunk for abstractive summarization"""
    inputs = tokenizer(
        "summarize this video transcript: " + text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=2,  # Reduced for speed
            length_penalty=1.2,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Add after app initialization
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add these near the top of the file with other global variables
gTTS_LOCK = Lock()
LAST_TTS_REQUEST = 0
MIN_TTS_DELAY = 1  # Minimum delay in seconds between TTS requests

def safe_tts_request(text, lang_code):
    """Helper function to handle TTS requests with rate limiting"""
    global LAST_TTS_REQUEST
    
    with gTTS_LOCK:
        # Calculate time since last request
        current_time = time.time()
        time_since_last = current_time - LAST_TTS_REQUEST
        
        # If needed, wait until enough time has passed
        if time_since_last < MIN_TTS_DELAY:
            time.sleep(MIN_TTS_DELAY - time_since_last)
        
        # Make TTS request
        tts = gTTS(text=text, lang=lang_code, slow=False)
        LAST_TTS_REQUEST = time.time()
        return tts

@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        logger.info("Starting summarization request")
        data = request.json
        
        # Start transcript fetch early
        video_id = data['url'].split("=")[1]
        logger.info(f"Fetching transcript for video ID: {video_id}")
        transcript_future = ThreadPoolExecutor().submit(get_cached_transcript, video_id)
        
        # Get other parameters while transcript is being fetched
        summary_type = data['summaryType']
        language = data['language']
        length_percentage = float(data.get('lengthPercentage', 0.3))
        
        # Get transcript result
        logger.info("Processing transcript")
        transcript = transcript_future.result()
        
        # Process transcript
        text = " ".join(line['text'] for line in transcript 
                       if len(line['text'].strip()) > 0)
        logger.info(f"Transcript length: {len(text.split())} words")

        # Generate summary
        logger.info(f"Generating {summary_type} summary")
        if summary_type == "Extractive":
            summary = extractive_summarize(text, length_percentage=length_percentage)
            logger.info(f"Extractive summary generated ({length_percentage*100}% length)")
        else:
            suggested_length = min(max(len(text.split()) // 8, 100), 150)
            summary = abstractive_summarize(
                text,
                max_length=suggested_length,
                min_length=min(suggested_length // 2, 50)
            )
            logger.info(f"Abstractive summary generated (target: {suggested_length} words)")

        # Translate if needed
        if language != "English":
            try:
                lang_code = LANGUAGE_CODES.get(language)
                if lang_code:
                    logger.info(f"Translating to {language} ({lang_code})")
                    summary = translate_text(summary, lang_code)
                    logger.info("Translation completed")
            except Exception as e:
                logger.error(f"Translation error: {str(e)}")

        # Generate audio
        audio_url = None
        try:
            lang_code = LANGUAGE_CODES.get(language, 'en')
            audio_filename = f'summary_{video_id}_{lang_code}.mp3'
            audio_path = os.path.join('static', audio_filename)
            
            # Remove existing file if it exists
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Removed existing audio file: {audio_path}")
            
            logger.info("Starting audio generation...")
            
            # Check if summary is empty or too long
            if not summary or len(summary) == 0:
                raise ValueError("Summary text is empty")
            
            if len(summary) > 5000:
                logger.info(f"Long summary detected ({len(summary)} chars), splitting into chunks")
                # Split long text into chunks
                chunks = [summary[i:i+5000] for i in range(0, len(summary), 5000)]
                
                temp_files = []
                for i, chunk in enumerate(chunks):
                    temp_filename = f'temp_{i}_{audio_filename}'
                    temp_path = os.path.join('static', temp_filename)
                    logger.info(f"Generating chunk {i+1}/{len(chunks)}")
                    
                    tts = safe_tts_request(chunk, lang_code)  # Use the new function
                    tts.save(temp_path)
                    temp_files.append(temp_path)
                    logger.info(f"Chunk {i+1} generated")
                
                # Combine all audio files
                logger.info("Combining audio chunks...")
                from pydub import AudioSegment
                combined = AudioSegment.empty()
                for temp_file in temp_files:
                    audio_segment = AudioSegment.from_mp3(temp_file)
                    combined += audio_segment
                
                combined.export(audio_path, format="mp3")
                
                # Clean up temp files
                for temp_file in temp_files:
                    os.remove(temp_file)
                
            else:
                logger.info(f"Generating audio for text length: {len(summary)} chars")
                tts = safe_tts_request(summary, lang_code)  # Use the new function
                tts.save(audio_path)
            
            audio_url = f"/static/{audio_filename}"
            logger.info(f"Audio generation completed: {audio_path}")

        except Exception as e:
            logger.error(f"Audio generation error: {str(e)}")
            logger.error(f"Error details:", exc_info=True)  # This will log the full traceback
            audio_url = None

        logger.info("Request completed successfully")
        return jsonify({
            "success": True,
            "summary": summary,
            "audio_url": audio_url,
            "original_text": text
        })

    except Exception as e:
        logger.error(f"Error in summarize: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (audio files)"""
    return send_from_directory('static', filename)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

def cleanup_old_files():
    """Clean up old audio files"""
    try:
        static_dir = 'static'
        current_time = time.time()
        for file in os.listdir(static_dir):
            if file.startswith('summary_') and file.endswith('.mp3'):
                file_path = os.path.join(static_dir, file)
                # Remove files older than 1 hour
                if current_time - os.path.getmtime(file_path) > 3600:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {file}")
    except Exception as e:
        print(f"Cleanup error: {str(e)}")

if __name__ == '__main__':
    from waitress import serve
    import atexit
    
    # Register cleanup function
    atexit.register(cleanup_old_files)
    
    # Use waitress for production-grade serving
    print("Server starting on http://localhost:8000")
    serve(app, host="0.0.0.0", port=8000)