from auth_and_cache import ( signup_user, login_user, logout_user, ContextCache )
import streamlit as st
import os
import time
import tempfile
import json
import re
import numpy as np
import faiss
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import hashlib
import io
import shutil
import yaml
from yaml.loader import SafeLoader

# Core Libraries
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Google GenAI library not found. Install with: pip install google-genai")
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Sentence Transformers not found. Install with: pip install sentence-transformers")
    st.stop()

try:
    import PyPDF2
except ImportError:
    st.error("PyPDF2 not found. Install with: pip install PyPDF2")
    st.stop()

# Video Processing Libraries (FFmpeg-free alternatives)
try:
    from faster_whisper import WhisperModel
except ImportError:
    st.error("faster-whisper not found. Install with: pip install faster-whisper")
    st.stop()

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    st.error("MoviePy not found. Install with: pip install moviepy")
    st.stop()

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
except ImportError:
    st.error("YouTube Transcript API not found. Install with: pip install youtube-transcript-api")
    st.stop()

# Better YouTube downloader - more reliable than pytube
try:
    import yt_dlp
except ImportError:
    st.error("yt-dlp not found. Install with: pip install yt-dlp")
    st.stop()

try:
    import streamlit_authenticator as stauth
except ImportError:
    st.error("Streamlit Authenticator not found. Install with: pip install streamlit-authenticator")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced RAG Chatbot - Multi-User & Video Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create user configuration with proper password hashing
def create_user_config():
    """Create user configuration with hashed passwords"""
    # Create credentials dictionary
    credentials = {
        "usernames": {
            "student1": {
                "email": "alice@student.edu",
                "name": "Alice Johnson",
                "password": "student123",  # Will be hashed automatically
                "role": "student"
            },
            "teacher1": {
                "email": "smith@university.edu",
                "name": "Prof. Smith",
                "password": "teacher123",  # Will be hashed automatically
                "role": "teacher"
            },
            "parent1": {
                "email": "wilson@parent.com",
                "name": "Mrs. Wilson",
                "password": "parent123",  # Will be hashed automatically
                "role": "parent"
            }
        }
    }
    
    # Hash passwords
    stauth.Hasher.hash_passwords(credentials)
    
    return {
        "credentials": credentials,
        "cookie": {
            "name": "rag_chatbot_cookie",
            "key": "random_signature_key_2024_advanced",
            "expiry_days": 30
        }
    }

# Initialize session state
session_defaults = {
    'authenticated': False,
    'api_key': None,
    'messages': [],
    'vector_store': None,
    'documents': [],
    'embeddings_model': None,
    'gemini_client': None,
    'grounding_threshold': 0.7,
    'whisper_model': None,
    'user_role': None,
    'username': None,
    'name': None,
    'authentication_status': None,
    'authenticator': None,  # Added for proper logout
    'logout_clicked': False  # Added to track logout action
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Role-based theming
def apply_role_theme(role):
    """Apply custom CSS based on user role"""
    themes = {
        'student': {
            'primary': '#1E88E5',
            'secondary': '#42A5F5',
            'accent': '#E3F2FD',
            'text': '#0D47A1'
        },
        'teacher': {
            'primary': '#43A047',
            'secondary': '#66BB6A',
            'accent': '#E8F5E8',
            'text': '#1B5E20'
        },
        'parent': {
            'primary': '#FB8C00',
            'secondary': '#FFB74D',
            'accent': '#FFF3E0',
            'text': '#E65100'
        }
    }
    
    if role in themes:
        theme = themes[role]
        st.markdown(f"""
        <style>
        .stButton > button {{
            background-color: {theme['primary']};
            color: white;
        }}
        .stSelectbox > div > div {{
            background-color: {theme['accent']};
        }}
        </style>
        """, unsafe_allow_html=True)

class VideoProcessor:
    """Handles video processing and audio extraction without FFmpeg"""
    
    def __init__(self):
        self.supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv']
    
    def extract_audio_from_video(self, video_file, output_path=None):
        """Extract audio from video using MoviePy (no FFmpeg required)"""
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.wav')
            
            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_file.read())
                temp_video_path = temp_video.name
            
            # Extract audio using MoviePy
            video_clip = VideoFileClip(temp_video_path)
            audio_clip = video_clip.audio
            
            # Write audio to WAV file
            audio_clip.write_audiofile(output_path, verbose=False, logger=None)
            
            # Clean up
            audio_clip.close()
            video_clip.close()
            os.unlink(temp_video_path)
            
            return output_path
            
        except Exception as e:
            st.error(f"Error extracting audio from video: {str(e)}")
            return None
    
    def get_youtube_transcript(self, url):
        """Get YouTube transcript using youtube-transcript-api with better error handling"""
        try:
            # Extract video ID from URL - improved regex
            if 'youtube.com/watch?v=' in url:
                video_id = url.split('watch?v=')[1].split('&')[0]
            elif 'youtu.be/' in url:
                video_id = url.split('youtu.be/')[1].split('?')[0]
            elif 'youtube.com/embed/' in url:
                video_id = url.split('embed/')[1].split('?')[0]
            else:
                # Try to extract from the end of the URL
                video_id = url.split('/')[-1].split('?')[0]
            
            # Try to get transcript with multiple language options
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                transcript_text = ' '.join([item['text'] for item in transcript_list])
                return transcript_text
            except NoTranscriptFound:
                # Try with auto-generated captions
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en-GB'])
                    transcript_text = ' '.join([item['text'] for item in transcript_list])
                    return transcript_text
                except:
                    pass
            
            return None
            
        except Exception as e:
            st.warning(f"Could not get YouTube transcript: {str(e)}")
            return None
    
    def download_youtube_audio(self, url):
        """### FFMPEG_FREE - Download YouTube audio without ffmpeg/ffprobe"""
        try:
            # Use completely ffmpeg-free approach
            temp_path = tempfile.mktemp(suffix='.m4a')
            
            # Configure yt-dlp to avoid any post-processing that requires ffmpeg
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',  # Prefer m4a, fallback to best
                'outtmpl': temp_path,
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'postprocessors': [],  # No post-processing to avoid ffmpeg dependency
                'prefer_ffmpeg': False,
                'merge_output_format': None,  # Don't merge
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignoreerrors': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check if file was created
            if os.path.exists(temp_path):
                return temp_path
            else:
                # Try alternative format if m4a fails
                temp_path2 = tempfile.mktemp(suffix='.webm')
                ydl_opts2 = {
                    'format': 'bestaudio[ext=webm]/bestaudio',
                    'outtmpl': temp_path2,
                    'quiet': True,
                    'no_warnings': True,
                    'noplaylist': True,
                    'postprocessors': [],
                    'prefer_ffmpeg': False,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts2) as ydl:
                    ydl.download([url])
                
                return temp_path2 if os.path.exists(temp_path2) else None
                
        except Exception as e:
            st.error(f"Error downloading YouTube audio: {str(e)}")
            return None

class WhisperTranscriber:
    """Handles transcription using faster-whisper (no FFmpeg required)"""
    
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
    
    def load_model(self):
        """Load WhisperModel - cached to avoid reloading"""
        if self.model is None:
            try:
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except Exception as e:
                st.error(f"Error loading Whisper model: {str(e)}")
                return None
        return self.model
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file using faster-whisper"""
        try:
            model = self.load_model()
            if model is None:
                return None
            
            segments, info = model.transcribe(audio_path, beam_size=5)
            
            # Combine all segments into full text
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "
            
            return full_text.strip()
            
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None

class GroundingValidator:
    """Validates if responses are properly grounded in provided context"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.min_overlap_threshold = 0.2
        self.semantic_threshold = 0.4
    
    def calculate_text_overlap(self, response: str, context: str) -> float:
        """Calculate overlap between response and context"""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        if not response_words:
            return 0.0
        
        overlap = len(response_words.intersection(context_words))
        return overlap / len(response_words)
    
    def calculate_semantic_similarity(self, response: str, context: str) -> float:
        """Calculate semantic similarity between response and context"""
        try:
            if not response.strip() or not context.strip():
                return 0.0
            
            response_embedding = self.embedding_model.encode([response])
            context_embedding = self.embedding_model.encode([context])
            
            # Calculate cosine similarity
            similarity = np.dot(response_embedding[0], context_embedding[0]) / (
                np.linalg.norm(response_embedding[0]) * np.linalg.norm(context_embedding[0])
            )
            
            return float(similarity)
            
        except Exception as e:
            st.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def validate_grounding(self, response: str, context: str) -> Dict[str, Any]:
        """Validate if response is properly grounded in context"""
        if not context.strip():
            return {
                'is_grounded': False,
                'confidence': 0.0,
                'reason': 'No context provided',
                'text_overlap': 0.0,
                'semantic_similarity': 0.0
            }
        
        # Calculate grounding metrics
        text_overlap = self.calculate_text_overlap(response, context)
        semantic_similarity = self.calculate_semantic_similarity(response, context)
        
        # Combined confidence score
        confidence = (text_overlap * 0.4) + (semantic_similarity * 0.6)
        
        # Determine if grounded
        is_grounded = (
            text_overlap >= self.min_overlap_threshold and
            semantic_similarity >= self.semantic_threshold
        )
        
        return {
            'is_grounded': is_grounded,
            'confidence': confidence,
            'reason': self._get_grounding_reason(text_overlap, semantic_similarity),
            'text_overlap': text_overlap,
            'semantic_similarity': semantic_similarity
        }
    
    def _get_grounding_reason(self, text_overlap: float, semantic_similarity: float) -> str:
        """Get reason for grounding decision"""
        if text_overlap < self.min_overlap_threshold:
            return f"Low text overlap ({text_overlap:.2f} < {self.min_overlap_threshold})"
        elif semantic_similarity < self.semantic_threshold:
            return f"Low semantic similarity ({semantic_similarity:.2f} < {self.semantic_threshold})"
        else:
            return "Well grounded in provided context"

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.transcriber = WhisperTranscriber()
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return ""
    
    def extract_text_from_video(self, video_file) -> str:
        """Extract text from video file using faster-whisper"""
        try:
            with st.spinner("Extracting audio from video..."):
                # Extract audio from video
                audio_path = self.video_processor.extract_audio_from_video(video_file)
                
                if audio_path:
                    with st.spinner("Transcribing audio to text..."):
                        # Transcribe audio to text
                        text = self.transcriber.transcribe_audio(audio_path)
                        # Clean up temporary audio file
                        os.unlink(audio_path)
                        return text if text else ""
                else:
                    return ""
                    
        except Exception as e:
            st.error(f"Error extracting text from video: {str(e)}")
            return ""
    
    def extract_text_from_youtube(self, youtube_url) -> str:
        """Extract text from YouTube video with improved error handling"""
        try:
            # First try to get transcript directly
            transcript = self.video_processor.get_youtube_transcript(youtube_url)
            if transcript:
                return transcript
            
            # If no transcript, download audio and transcribe
            with st.spinner("Downloading YouTube audio..."):
                audio_path = self.video_processor.download_youtube_audio(youtube_url)
                
                if audio_path:
                    with st.spinner("Transcribing YouTube audio..."):
                        text = self.transcriber.transcribe_audio(audio_path)
                        # Clean up temporary audio file
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
                        return text if text else ""
                else:
                    return ""
                    
        except Exception as e:
            st.error(f"Error extracting text from YouTube: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap - optimized for grounding"""
        chunks = []
        
        # Split by paragraphs first for better context preservation
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If chunks are too large, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                words = chunk.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) < chunk_size:
                        current_chunk += word + " "
                    else:
                        if current_chunk.strip():
                            final_chunks.append(current_chunk.strip())
                        current_chunk = word + " "
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks

class RAGVectorStore:
    """Enhanced vector storage with better relevance scoring"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.embeddings = None
        self.document_metadata = []
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to vector store with metadata and context caching"""
        try:
            # Initialize context cache if not exists
            if not hasattr(st.session_state, 'context_cache'):
                st.session_state.context_cache = ContextCache(self.embedding_model)
            
            # Apply context caching to documents
            cached_documents = []
            for doc in documents:
                cached_doc = st.session_state.context_cache.get_chunk(doc)
                cached_documents.append(cached_doc)
            
            with st.spinner("Creating embeddings..."):
                # Generate embeddings for cached documents
                embeddings = self.embedding_model.encode(cached_documents, show_progress_bar=False)
                
                if self.index is None:
                    # Create FAISS index
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    self.embeddings = embeddings
                    self.documents = cached_documents
                    self.document_metadata = metadata or [{}] * len(cached_documents)
                else:
                    # Add to existing index
                    self.embeddings = np.vstack([self.embeddings, embeddings])
                    self.documents.extend(cached_documents)
                    self.document_metadata.extend(metadata or [{}] * len(cached_documents))
                
                # Add embeddings to index
                self.index.add(embeddings.astype('float32'))
                
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5, relevance_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Enhanced search with relevance filtering"""
        try:
            if self.index is None:
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in FAISS index with more candidates
            search_k = min(k * 2, len(self.documents))
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    # Convert distance to relevance score
                    relevance_score = 1.0 / (1.0 + distance)
                    
                    # Only include results above relevance threshold
                    if relevance_score >= relevance_threshold:
                        results.append({
                            'content': self.documents[idx],
                            'distance': float(distance),
                            'relevance_score': relevance_score,
                            'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                        })
            
            # Sort by relevance score and return top k
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []

class GroundedGeminiChatbot:
    """Enhanced RAG Chatbot with strict grounding and extended responses"""
    
    def __init__(self, api_key: str, grounding_validator: GroundingValidator):
        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.0-flash"
            self.grounding_validator = grounding_validator
            self.max_retries = 2
            st.session_state.gemini_client = self.client
        except Exception as e:
            st.error(f"Error initializing Gemini client: {str(e)}")
            self.client = None
    
    def _create_grounded_prompt(self, query: str, context: str) -> str:
        """Create a strictly grounded prompt for detailed responses"""
        if not context.strip():
            return self._create_no_context_prompt(query)
        
        # Enhanced system prompt for strict grounding with detailed responses
        grounded_prompt = f"""You are a helpful assistant that MUST answer questions based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. You can ONLY use information that is explicitly stated in the context below
2. Do NOT use any external knowledge or information not in the context
3. If the context doesn't contain enough information to answer the question, you MUST say "I don't have enough information in the provided context to answer this question"
4. Quote relevant parts of the context when possible
5. Stay strictly within the bounds of the provided information
6. Provide comprehensive, detailed answers when the context allows
7. Use proper formatting with headers, bullet points, and structure when appropriate
8. Include all relevant details from the context
9. Organize your response logically with clear sections

CONTEXT:
{context}

QUESTION: {query}

REQUIREMENTS:
- Base your answer ONLY on the context above
- If information is not in the context, explicitly state that you don't have that information
- Include specific quotes or references from the context to support your answer
- Do not make assumptions or add information not present in the context
- Provide as much detail as possible from the available context
- Structure your answer clearly with proper formatting
- Use markdown formatting for better readability (headers, lists, bold text)
- Be comprehensive and thorough in your response

ANSWER:"""
        
        return grounded_prompt
    
    def _create_no_context_prompt(self, query: str) -> str:
        """Create prompt when no context is available"""
        return f"""I don't have any relevant information in my knowledge base to answer your question: "{query}"

Please try:
1. Uploading relevant documents that contain the information you're looking for
2. Rephrasing your question to be more specific
3. Asking a question that relates to the documents you've uploaded

I can only provide answers based on the documents you've provided to me."""
    
    def _validate_and_improve_response(self, response: str, context: str, query: str) -> Dict[str, Any]:
        """Validate response grounding and improve if needed"""
        # Check grounding
        grounding_result = self.grounding_validator.validate_grounding(response, context)
        
        # If not well grounded, provide fallback response
        if not grounding_result['is_grounded']:
            fallback_response = self._generate_fallback_response(query, context)
            return {
                'response': fallback_response,
                'grounding_result': grounding_result,
                'is_fallback': True
            }
        
        return {
            'response': response,
            'grounding_result': grounding_result,
            'is_fallback': False
        }
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when grounding fails"""
        if not context.strip():
            return f"I don't have any relevant information in my knowledge base to answer your question about '{query}'. Please upload relevant documents first."
        
        return f"I cannot provide a complete answer to your question about '{query}' based on the available context. The information in my knowledge base may not be sufficient or directly relevant to your specific question. Please try rephrasing your question or providing more specific documents."
    
    def generate_response(self, query: str, context: str = "", conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate grounded response with validation - EXTENDED LENGTH (4096 tokens)"""
        try:
            if not self.client:
                return {
                    'response': "Error: Gemini client not initialized.",
                    'grounding_result': None,
                    'is_fallback': True
                }
            
            # Create grounded prompt
            prompt = self._create_grounded_prompt(query, context)
            
            # Generate response with enhanced configuration - INCREASED TOKEN LIMIT
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Lower temperature for more deterministic responses
                    top_p=0.8,
                    max_output_tokens=4096,  # INCREASED FROM 1024 TO 4096 FOR LONGER DETAILED ANSWERS
                    stop_sequences=["EXTERNAL:", "OUTSIDE:", "GENERAL KNOWLEDGE:"]
                )
            )
            
            # Validate and improve response
            result = self._validate_and_improve_response(response.text, context, query)
            return result
            
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'grounding_result': None,
                'is_fallback': True
            }

def authenticate_user():
    """Fixed authentication function for latest streamlit-authenticator"""
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown("### 🔐 Login to Digital Companion")
    
    # Create user config
    user_config = create_user_config()
    
    # Initialize authenticator
    authenticator = stauth.Authenticate(
        user_config["credentials"],
        user_config["cookie"]["name"],
        user_config["cookie"]["key"],
        user_config["cookie"]["expiry_days"]
    )
    
    # Store authenticator in session state
    st.session_state.authenticator = authenticator
    
    # Login form
    try:
        name, authentication_status, username = authenticator.login()
        
        if authentication_status == True:
            st.session_state.authenticated = True
            st.session_state.name = name
            st.session_state.username = username
            # Get user role from credentials
            st.session_state.user_role = user_config["credentials"]["usernames"][username]["role"]
            st.success(f"Welcome {name}!")
            st.rerun()
            
        elif authentication_status == False:
            st.error("Username/password is incorrect")
            
        elif authentication_status == None:
            st.warning("Please enter your username and password")
            
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def initialize_models():
    """Initialize embedding models and other components"""
    try:
        if st.session_state.embeddings_model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Embedding model loaded successfully!")
        
        # Initialize vector store if not exists
        if st.session_state.vector_store is None:
            st.session_state.vector_store = RAGVectorStore(st.session_state.embeddings_model)
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return False

def configure_gemini():
    """Configure Gemini API"""
    st.markdown("### 🔑 Gemini API Configuration")
    
    # Check for API key in secrets or environment
    api_key = None
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
        st.success("✅ API key loaded from secrets!")
    elif 'GEMINI_API_KEY' in os.environ:
        api_key = os.environ['GEMINI_API_KEY']
        st.success("✅ API key loaded from environment!")
    else:
        st.warning("⚠️ No API key found in secrets or environment")
        api_key = st.text_input("Enter your Gemini API Key:", type="password")
    
    if api_key:
        st.session_state.api_key = api_key
        # Initialize grounding validator and chatbot
        if st.session_state.embeddings_model:
            grounding_validator = GroundingValidator(st.session_state.embeddings_model)
            st.session_state.chatbot = GroundedGeminiChatbot(api_key, grounding_validator)
        return True
    
    return False

def document_upload_interface():
    """Document upload and processing interface"""
    st.sidebar.markdown("### 📄 Document Upload")
    
    processor = DocumentProcessor()
    
    # File upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'mp4', 'avi', 'mov', 'mkv', 'webm'],
        accept_multiple_files=True,
        help="Upload PDF, TXT files, or video files for processing"
    )
    
    # YouTube URL input
    youtube_url = st.sidebar.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Process button
    if st.sidebar.button("🔄 Process Documents", type="primary"):
        if not uploaded_files and not youtube_url:
            st.sidebar.warning("Please upload files or provide a YouTube URL")
            return
        
        all_text = ""
        processed_count = 0
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            try:
                file_type = uploaded_file.type
                
                if file_type == "application/pdf":
                    text = processor.extract_text_from_pdf(uploaded_file)
                elif file_type == "text/plain":
                    text = processor.extract_text_from_txt(uploaded_file)
                elif file_type.startswith("video/"):
                    text = processor.extract_text_from_video(uploaded_file)
                else:
                    st.sidebar.warning(f"Unsupported file type: {file_type}")
                    continue
                
                if text.strip():
                    all_text += f"\n\n--- {uploaded_file.name} ---\n\n{text}"
                    processed_count += 1
                else:
                    st.sidebar.warning(f"No text extracted from {uploaded_file.name}")
                    
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Process YouTube URL
        if youtube_url:
            try:
                text = processor.extract_text_from_youtube(youtube_url)
                if text.strip():
                    all_text += f"\n\n--- YouTube: {youtube_url} ---\n\n{text}"
                    processed_count += 1
                else:
                    st.sidebar.warning("No text extracted from YouTube video")
            except Exception as e:
                st.sidebar.error(f"Error processing YouTube URL: {str(e)}")
        
        # Add to vector store
        if all_text.strip() and st.session_state.vector_store:
            try:
                # Chunk the text
                chunks = processor.chunk_text(all_text)
                
                # Add to vector store
                success = st.session_state.vector_store.add_documents(chunks)
                
                if success:
                    st.session_state.documents.extend(chunks)
                    st.sidebar.success(f"✅ Processed {processed_count} documents with {len(chunks)} chunks")
                else:
                    st.sidebar.error("Failed to add documents to vector store")
                    
            except Exception as e:
                st.sidebar.error(f"Error adding to vector store: {str(e)}")
        else:
            st.sidebar.warning("No text content to process")

def chat_interface():
    """Main chat interface - FIXED to avoid nested st.chat_input()"""
    
    # Display current role with theme
    role_display = {
        'student': '📚 Student Mode',
        'teacher': '👨‍🏫 Teacher Mode', 
        'parent': '👨‍👩‍👧‍👦 Parent Mode'
    }
    
    current_role = st.session_state.user_role
    st.markdown(f"### 📖 Chat with Your Knowledge Base - {role_display.get(current_role, 'Unknown Mode')}")
    
    # Apply theme
    apply_role_theme(current_role)
    
    # Display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # CRITICAL FIX: Move st.chat_input() to TOP LEVEL - not inside any containers
    prompt = st.chat_input("Ask me anything...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search for relevant context
                context = ""
                if st.session_state.vector_store and st.session_state.documents:
                    search_results = st.session_state.vector_store.search(prompt, k=5)
                    if search_results:
                        context = "\n\n".join([result['content'] for result in search_results])
                
                # Generate response using chatbot
                if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot:
                    result = st.session_state.chatbot.generate_response(prompt, context)
                    response = result['response']
                    
                    # Display grounding info if available
                    if result.get('grounding_result'):
                        grounding = result['grounding_result']
                        if grounding['is_grounded']:
                            st.caption(f"✅ Response grounded (confidence: {grounding['confidence']:.2f})")
                        else:
                            st.caption(f"⚠️ {grounding['reason']}")
                else:
                    response = "Please configure your Gemini API key first."
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

def sidebar_controls():
    """Sidebar controls and information"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Document count
    doc_count = len(st.session_state.documents) if st.session_state.documents else 0
    st.sidebar.metric("Documents Processed", doc_count)
    
    # Grounding threshold
    st.session_state.grounding_threshold = st.sidebar.slider(
        "Grounding Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.grounding_threshold,
        step=0.1,
        help="Higher values require stronger grounding"
    )
    
    # Clear documents
    if st.sidebar.button("🗑️ Clear All Documents"):
        st.session_state.documents = []
        st.session_state.vector_store = RAGVectorStore(st.session_state.embeddings_model) if st.session_state.embeddings_model else None
        st.sidebar.success("Documents cleared!")
    
    # Clear chat history
    if st.sidebar.button("💬 Clear Chat History"):
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared!")
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def main():
    """Main application function"""
    
    # Check authentication
    if not st.session_state.authenticated:
        authenticate_user()
        return
    
    # Show user info
    st.markdown(f"**Welcome, {st.session_state.name}!** ({st.session_state.user_role.title()})")
    
    # Initialize models
    if not initialize_models():
        st.error("Failed to initialize models. Please refresh the page.")
        return
    
    # Configure Gemini
    if not configure_gemini():
        st.warning("Please configure your Gemini API key to start chatting.")
        return
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main chat interface
        chat_interface()
    
    with col2:
        # Document upload interface
        document_upload_interface()
        
        # Sidebar controls
        sidebar_controls()
    
    # Footer
    st.markdown("---")
    role_footers = {
        'student': '🎓 Student Learning Assistant',
        'teacher': '📋 Teaching Support Tool',
        'parent': '👪 Family Education Helper'
    }
    
    st.markdown(f"""
    <div style='text-align: center; color: gray; padding: 20px;'>
    <strong>{role_footers.get(st.session_state.user_role, 'Advanced RAG Chatbot')}</strong><br>
    🤖 Powered by Gemini 2.0 Flash • 🎥 Video Support • 🔒 Multi-User Authentication<br>
    📝 Strict Grounding • 🚫 No FFmpeg Required • 🎯 Role-Based Access • 📤 Unlimited File Size<br>
    ✨ Extended Responses (4096 tokens) • 🚪 Auto-Redirect Logout • 📊 Enhanced Analytics
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()