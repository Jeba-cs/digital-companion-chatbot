from auth_and_cache import (
    signup_user,
    login_user,
    logout_user,
    ContextCache
)


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
        .stApp {{
            background: linear-gradient(135deg, {theme['accent']} 0%, #ffffff 100%);
        }}
        .stButton > button {{
            background: {theme['primary']};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }}
        .stButton > button:hover {{
            background: {theme['secondary']};
        }}
        .stSelectbox > div > div {{
            background: {theme['accent']};
        }}
        .role-header {{
            background: {theme['primary']};
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
        }}
        .metric-card {{
            background: {theme['accent']};
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid {theme['primary']};
            margin: 0.5rem 0;
        }}
        .logout-btn {{
            background: #dc3545 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
        }}
        .logout-btn:hover {{
            background: #c82333 !important;
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

    def generate_response(self, query: str, context: str = "", conversation_history: List[Dict] = None) -> Dict[
        str, Any]:
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
    st.markdown('<div class="role-header">🔐 Advanced RAG Chatbot - Multi-User Login</div>', unsafe_allow_html=True)

    # Get user configuration
    config = create_user_config()

    # Create authenticator with corrected parameters
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # ### AUTH_LOGOUT - Store authenticator in session state for proper logout
    st.session_state.authenticator = authenticator

    # **FIXED LOGIN CALL** - Use new syntax without parameters
    try:
        authenticator.login()
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return

    # **FIXED SESSION STATE ACCESS** - Use new session state keys
    if st.session_state.get('authentication_status'):
        st.session_state.authenticated = True
        st.session_state.username = st.session_state.get('username')
        st.session_state.name = st.session_state.get('name')

        # Get user role from credentials
        user_data = config['credentials']['usernames'].get(st.session_state.username, {})
        st.session_state.user_role = user_data.get('role', 'student')

        # Apply role-based theme
        apply_role_theme(st.session_state.user_role)

        st.success(f"✅ Welcome {st.session_state.name}! Role: {st.session_state.user_role.title()}")

        # Show role-specific welcome message
        role_messages = {
            'student': "📚 Access your learning materials and ask questions about uploaded content.",
            'teacher': "👨‍🏫 Manage educational content and track student interactions.",
            'parent': "👨‍👩‍👧‍👦 Review educational materials and monitor learning progress."
        }

        st.info(role_messages.get(st.session_state.user_role, "Welcome to the RAG Chatbot!"))

        # **FIXED LOGOUT CALL** - Use new syntax
        authenticator.logout()

        time.sleep(1)
        st.rerun()

    elif st.session_state.get('authentication_status') is False:
        st.error('❌ Username/password is incorrect')

    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')

    # Demo credentials info
    with st.expander("🔍 Demo Credentials"):
        st.info("""
        **Demo Accounts:**
        - Student: username=`student1`, password=`student123`
        - Teacher: username=`teacher1`, password=`teacher123`  
        - Parent: username=`parent1`, password=`parent123`
        """)


def initialize_models():
    """Initialize embedding model and RAG components"""
    if st.session_state.embeddings_model is None:
        with st.spinner("Loading embedding model..."):
            try:
                st.session_state.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Embedding model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return False

    if st.session_state.vector_store is None:
        st.session_state.vector_store = RAGVectorStore(st.session_state.embeddings_model)

    return True


def get_api_key():
    """FIXED: Get API key from user with proper error handling"""
    st.subheader("🔑 Gemini API Configuration")

    # **FIXED: Try to get from secrets with proper error handling**
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            st.success("✅ API key loaded from secrets!")
    except Exception as e:
        # Secrets file not found or key not in secrets - this is expected
        st.info("💡 No secrets file found. Please enter your API key manually.")

    if not api_key:
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            placeholder="Your API key here...",
            help="Get your API key from Google AI Studio"
        )

    if api_key:
        st.session_state.api_key = api_key
        return True

    st.warning("Please enter your Gemini API key to continue.")
    st.info("""
    **To set up automatic API key loading:**
    1. Create folder: `.streamlit` in your project directory
    2. Create file: `secrets.toml` in the `.streamlit` folder
    3. Add your key: `GEMINI_API_KEY = "your_api_key_here"`
    4. Restart the app
    """)
    return False


def document_upload_section():
    """Enhanced document upload with video support and updated student limits"""
    st.sidebar.header("📁 Knowledge Base Management")

    # **UPDATED: Modified student limits as requested**
    upload_limits = {
        'student': {'files': 10},  # Changed from 5 files and removed size limit
        'teacher': {'files': 20, 'size': '50MB'},
        'parent': {'files': 10, 'size': '25MB'}
    }

    role = st.session_state.user_role
    limit = upload_limits.get(role, {'files': 10})

    # **UPDATED: Display new limits**
    if role == 'student':
        st.sidebar.info(f"📊 **{role.title()} Limits:**\n- Max files: {limit['files']}")
    else:
        st.sidebar.info(f"📊 **{role.title()} Limits:**\n- Max files: {limit['files']}\n- Max size: {limit['size']}")

    # File upload tabs
    tab1, tab2, tab3 = st.sidebar.tabs(["📄 Documents", "🎥 Videos", "🌐 YouTube"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            help="Upload PDF or TXT files"
        )

        if uploaded_files and st.button("🔄 Process Documents", key="process_docs"):
            process_documents(uploaded_files)

    with tab2:
        uploaded_videos = st.file_uploader(
            "Upload Videos",
            accept_multiple_files=True,
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload video files for transcription"
        )

        if uploaded_videos and st.button("🎬 Process Videos", key="process_videos"):
            process_videos(uploaded_videos)

    with tab3:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter YouTube video URL"
        )

        if youtube_url and st.button("📺 Process YouTube", key="process_youtube"):
            process_youtube(youtube_url)

    # Display current knowledge base
    if st.session_state.documents:
        st.sidebar.subheader("📚 Current Knowledge Base")
        st.sidebar.info(f"Total chunks: {len(st.session_state.documents)}")

        # Grounding settings
        st.sidebar.subheader("🎯 Grounding Settings")
        st.session_state.grounding_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values require stronger grounding"
        )

        if st.sidebar.button("🗑️ Clear Knowledge Base"):
            st.session_state.documents = []
            st.session_state.vector_store = RAGVectorStore(st.session_state.embeddings_model)
            st.sidebar.success("Knowledge base cleared!")
            st.rerun()


def process_documents(uploaded_files):
    """Process uploaded documents"""
    processor = DocumentProcessor()
    all_chunks = []
    metadata = []

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")

        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = processor.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = processor.extract_text_from_txt(uploaded_file)
        else:
            st.sidebar.error(f"Unsupported file type: {uploaded_file.type}")
            continue

        if text:
            chunks = processor.chunk_text(text, chunk_size=400, overlap=50)
            all_chunks.extend(chunks)

            for chunk in chunks:
                metadata.append({
                    'source_file': uploaded_file.name,
                    'source_type': 'document',
                    'chunk_length': len(chunk),
                    'processing_time': datetime.now().isoformat(),
                    'processed_by': st.session_state.username
                })

        progress_bar.progress((i + 1) / len(uploaded_files))

    if all_chunks:
        if st.session_state.vector_store.add_documents(all_chunks, metadata):
            st.session_state.documents.extend(all_chunks)
            st.sidebar.success(f"✅ Processed {len(all_chunks)} document chunks!")
        else:
            st.sidebar.error("❌ Failed to process documents")

    status_text.empty()
    progress_bar.empty()


def process_videos(uploaded_videos):
    """Process uploaded video files"""
    processor = DocumentProcessor()
    all_chunks = []
    metadata = []

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for i, uploaded_video in enumerate(uploaded_videos):
        status_text.text(f"Processing {uploaded_video.name}...")

        # Extract text from video
        text = processor.extract_text_from_video(uploaded_video)

        if text:
            chunks = processor.chunk_text(text, chunk_size=400, overlap=50)
            all_chunks.extend(chunks)

            for chunk in chunks:
                metadata.append({
                    'source_file': uploaded_video.name,
                    'source_type': 'video',
                    'chunk_length': len(chunk),
                    'processing_time': datetime.now().isoformat(),
                    'processed_by': st.session_state.username
                })

        progress_bar.progress((i + 1) / len(uploaded_videos))

    if all_chunks:
        if st.session_state.vector_store.add_documents(all_chunks, metadata):
            st.session_state.documents.extend(all_chunks)
            st.sidebar.success(f"✅ Processed {len(all_chunks)} video chunks!")
        else:
            st.sidebar.error("❌ Failed to process videos")

    status_text.empty()
    progress_bar.empty()


def process_youtube(youtube_url):
    """FIXED: Process YouTube video with proper spinner usage and better error handling"""
    processor = DocumentProcessor()

    # **FIXED: Use with st.sidebar: and then st.spinner() inside**
    with st.sidebar:
        with st.spinner("Processing YouTube video..."):
            text = processor.extract_text_from_youtube(youtube_url)

            if text:
                chunks = processor.chunk_text(text, chunk_size=400, overlap=50)
                metadata = []

                for chunk in chunks:
                    metadata.append({
                        'source_file': youtube_url,
                        'source_type': 'youtube',
                        'chunk_length': len(chunk),
                        'processing_time': datetime.now().isoformat(),
                        'processed_by': st.session_state.username
                    })

                if st.session_state.vector_store.add_documents(chunks, metadata):
                    st.session_state.documents.extend(chunks)
                    st.sidebar.success(f"✅ Processed {len(chunks)} YouTube chunks!")
                else:
                    st.sidebar.error("❌ Failed to process YouTube video")
            else:
                st.sidebar.error("❌ Could not extract text from YouTube video")


def chat_interface():
    """Enhanced chat interface with role-based features and extended responses"""
    role = st.session_state.user_role

    # Role-based header
    role_emojis = {'student': '📚', 'teacher': '👨‍🏫', 'parent': '👨‍👩‍👧‍👦'}
    emoji = role_emojis.get(role, '🤖')

    st.header(f"{emoji} Chat with Your Knowledge Base - {role.title()} Mode")

    # Initialize components
    if 'grounding_validator' not in st.session_state:
        st.session_state.grounding_validator = GroundingValidator(st.session_state.embeddings_model)

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = GroundedGeminiChatbot(
            st.session_state.api_key,
            st.session_state.grounding_validator
        )

    # Display chat messages with enhanced formatting for longer responses
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Enhanced markdown rendering for longer, detailed responses
            st.markdown(message["content"], unsafe_allow_html=True)

            if message["role"] == "assistant":
                # Show grounding information
                if "grounding_result" in message and message["grounding_result"]:
                    grounding = message["grounding_result"]

                    # Color code based on grounding quality
                    if grounding['confidence'] >= 0.8:
                        confidence_color = "🟢"
                    elif grounding['confidence'] >= 0.6:
                        confidence_color = "🟡"
                    else:
                        confidence_color = "🔴"

                    st.markdown(f"{confidence_color} **Grounding Confidence:** {grounding['confidence']:.2f}")

                    with st.expander("🔍 Grounding Details"):
                        st.write(f"**Well Grounded:** {'Yes' if grounding['is_grounded'] else 'No'}")
                        st.write(f"**Text Overlap:** {grounding['text_overlap']:.2f}")
                        st.write(f"**Semantic Similarity:** {grounding['semantic_similarity']:.2f}")
                        st.write(f"**Reason:** {grounding['reason']}")

                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i + 1}** (Relevance: {source['relevance_score']:.2f})")
                            st.markdown(f"*Type:* {source['metadata'].get('source_type', 'Unknown')}")
                            st.markdown(f"*File:* {source['metadata'].get('source_file', 'Unknown')}")
                            st.code(
                                source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])

    # Role-based chat input prompts
    prompts = {
        'student': "Ask detailed questions about your study materials...",
        'teacher': "Query educational content for comprehensive analysis...",
        'parent': "Ask about learning materials and educational content..."
    }

    # Chat input
    if prompt := st.chat_input(prompts.get(role, "Ask me anything...")):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating comprehensive response..."):
                # Search for relevant documents with extended context
                context = ""
                sources = []

                if st.session_state.vector_store and st.session_state.documents:
                    search_results = st.session_state.vector_store.search(
                        prompt,
                        k=5,  # Increased from 3 to 5 for more comprehensive context
                        relevance_threshold=0.3
                    )
                    if search_results:
                        context = "\n\n".join([result["content"] for result in search_results])
                        sources = search_results

                # Generate grounded response with extended length capability
                response_data = st.session_state.chatbot.generate_response(
                    prompt,
                    context,
                    st.session_state.messages
                )

                response = response_data['response']
                grounding_result = response_data['grounding_result']
                is_fallback = response_data['is_fallback']

                # Display response with enhanced formatting
                st.markdown(response, unsafe_allow_html=True)

                # Show grounding information
                if grounding_result:
                    # Color code based on grounding quality
                    if grounding_result['confidence'] >= 0.8:
                        confidence_color = "🟢"
                    elif grounding_result['confidence'] >= 0.6:
                        confidence_color = "🟡"
                    else:
                        confidence_color = "🔴"

                    st.markdown(f"{confidence_color} **Grounding Confidence:** {grounding_result['confidence']:.2f}")

                    with st.expander("🔍 Grounding Details"):
                        st.write(f"**Well Grounded:** {'Yes' if grounding_result['is_grounded'] else 'No'}")
                        st.write(f"**Text Overlap:** {grounding_result['text_overlap']:.2f}")
                        st.write(f"**Semantic Similarity:** {grounding_result['semantic_similarity']:.2f}")
                        st.write(f"**Reason:** {grounding_result['reason']}")
                        if is_fallback:
                            st.warning("⚠️ Fallback response used due to poor grounding")

                # Display sources if available
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i + 1}** (Relevance: {source['relevance_score']:.2f})")
                            st.markdown(f"*Type:* {source['metadata'].get('source_type', 'Unknown')}")
                            st.markdown(f"*File:* {source['metadata'].get('source_file', 'Unknown')}")
                            st.code(
                                source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])

        # Add assistant message
        assistant_message = {
            "role": "assistant",
            "content": response,
            "grounding_result": grounding_result,
            "is_fallback": is_fallback
        }
        if sources:
            assistant_message["sources"] = sources
        st.session_state.messages.append(assistant_message)


def sidebar_controls():
    """Enhanced sidebar controls with WORKING AUTO-REDIRECT LOGOUT"""
    st.sidebar.header("⚙️ Settings")

    # ### FIXED LOGOUT - With automatic redirect to login page
    if (
            st.session_state.get("authenticated") and
            st.session_state.get("authentication_status") and
            st.session_state.get("authenticator")
    ):
        # Store previous authentication status to detect logout
        prev_auth_status = st.session_state.get("authentication_status")

        st.sidebar.markdown("---")
        # Use the official logout method from streamlit-authenticator
        st.session_state.authenticator.logout('🚪 Logout', 'sidebar')
        st.sidebar.markdown("---")

        # Check if user has been logged out and redirect to login page
        if prev_auth_status and not st.session_state.get("authentication_status"):
            # Clear all session state data
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.session_state.user_role = None

            # Show logout success message
            st.sidebar.success("✅ Successfully logged out!")

            # Force page refresh to show login screen
            time.sleep(0.5)  # Brief delay to show message
            st.rerun()

    # User info (only show if authenticated)
    if st.session_state.get("authenticated"):
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <strong>👤 User:</strong> {st.session_state.name}<br>
            <strong>🎭 Role:</strong> {st.session_state.user_role.title()}<br>
            <strong>📧 Session:</strong> Active
        </div>
        """, unsafe_allow_html=True)

    # Custom logout helper
    if st.sidebar.button("🚪 Quick Logout"):
        logout_user()
        for key in ['authenticated', 'username', 'name', 'user_role', 'authentication_status']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


    # Model information
    with st.sidebar.expander("🤖 Model Info"):
        st.info("""
        **Model:** Gemini 2.0 Flash
        **Max Tokens:** 4096 (Extended)
        **Embedding:** all-MiniLM-L6-v2
        **Vector Store:** FAISS
        **Grounding:** Strict Document-Only
        **Video:** faster-whisper (FFmpeg-free)
        **YouTube:** yt-dlp (FFmpeg-free)
        """)

    # Chat statistics (only show if authenticated and have messages)
    if st.session_state.get("authenticated") and st.session_state.messages:
        st.sidebar.header("📊 Chat Statistics")

        grounded_responses = 0
        total_responses = 0
        avg_confidence = 0

        for message in st.session_state.messages:
            if message["role"] == "assistant" and "grounding_result" in message:
                total_responses += 1
                if message["grounding_result"] and message["grounding_result"]["is_grounded"]:
                    grounded_responses += 1
                if message["grounding_result"]:
                    avg_confidence += message["grounding_result"]["confidence"]

        if total_responses > 0:
            grounding_rate = (grounded_responses / total_responses) * 100
            avg_confidence = avg_confidence / total_responses

            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Grounding Rate", f"{grounding_rate:.1f}%")
            with col2:
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")

            st.sidebar.metric("Total Messages", len(st.session_state.messages))

    # Controls (only show if authenticated)
    if st.session_state.get("authenticated"):
        st.sidebar.header("🎮 Controls")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("💾 Export Chat"):
                export_chat()


def export_chat():
    """Export chat history with enhanced metadata"""
    if st.session_state.messages:
        chat_data = {
            "export_timestamp": datetime.now().isoformat(),
            "user_info": {
                "username": st.session_state.username,
                "name": st.session_state.name,
                "role": st.session_state.user_role
            },
            "messages": st.session_state.messages,
            "settings": {
                "grounding_threshold": st.session_state.grounding_threshold,
                "total_documents": len(st.session_state.documents),
                "max_tokens": 4096
            },
            "session_stats": {
                "total_messages": len(st.session_state.messages),
                "document_count": len(st.session_state.documents)
            }
        }

        json_str = json.dumps(chat_data, indent=2)

        st.sidebar.download_button(
            label="📥 Download Chat",
            data=json_str,
            file_name=f"chat_export_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Main application logic"""

    # Apply role-based theme if authenticated
    if st.session_state.authenticated:
        apply_role_theme(st.session_state.user_role)

    # Check authentication - if not authenticated, show login and stop
    if not st.session_state.authenticated:
        choice = st.sidebar.radio("Choose action", ["Log In", "Sign Up"])
        if choice == "Sign Up":
            # Show signup form
            with st.form("signup_form", clear_on_submit=True):
                new_username = st.text_input("Username")
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email Address")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                role = st.selectbox("Role", ["student", "teacher", "parent"])
                submitted = st.form_submit_button("Sign Up")

            if submitted:
                try:
                    form_data = {
                        "username": new_username,
                        "name": new_name,
                        "email": new_email,
                        "password": new_password,
                        "confirm_password": confirm_password,
                        "role": role
                    }
                    if signup_user(form_data):
                        st.success("Account created! Please log in.")
                        st.experimental_rerun()
                except ValueError as e:
                    st.error(str(e))
        else:
            # Show login form
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Login")

            if login_submitted:
                try:
                    user_info = login_user(username, password)
                    st.session_state.authenticated = True
                    st.session_state.username = user_info["username"]
                    st.session_state.name = user_info["name"]
                    st.session_state.user_role = user_info["role"]
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        return

    # Initialize models
    if not initialize_models():
        st.error("Failed to initialize models. Please refresh the page.")
        return

    # Get API key
    if not get_api_key():
        return

    # Main interface
    col1, col2 = st.columns([3, 1])

    with col1:
        chat_interface()

    with col2:
        document_upload_section()
        sidebar_controls()

    # Role-based footer
    role_footers = {
        'student': '📚 Enhanced Learning Experience',
        'teacher': '🎓 Educational Content Management',
        'parent': '👨‍👩‍👧‍👦 Family Learning Support'
    }

    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎯 <strong>{role_footers.get(st.session_state.user_role, 'Advanced RAG Chatbot')}</strong></p>
        <p>🤖 Powered by Gemini 2.0 Flash • 🎥 Video Support • 🔒 Multi-User Authentication</p>
        <p>📝 Strict Grounding • 🚫 No FFmpeg Required • 🎯 Role-Based Access • 📤 Unlimited File Size</p>
        <p>✨ Extended Responses (4096 tokens) • 🚪 Auto-Redirect Logout • 📊 Enhanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
