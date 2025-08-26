# auth_and_cache.py

import os
import time
import yaml
import hashlib
import pickle
from typing import List, Dict, Any
from yaml.loader import SafeLoader
from sentence_transformers import SentenceTransformer
import numpy as np

# ─── USER AUTHENTICATION ─────────────────────────────────────────────────────────

USER_DB = "users.yaml"

def load_users() -> Dict[str, Any]:
    """Load user database from YAML file."""
    if not os.path.exists(USER_DB):
        with open(USER_DB, "w") as f:
            yaml.dump({"usernames": {}}, f)
    with open(USER_DB) as f:
        return yaml.load(f, Loader=SafeLoader)

def save_users(db: Dict[str, Any]):
    """Save user database to YAML file."""
    with open(USER_DB, "w") as f:
        yaml.dump(db, f)

def signup_user(form_data: Dict[str, str]) -> bool:
    """
    Register a new user.
    form_data keys: username, name, email, password, confirm_password, role.
    Raises ValueError on validation failure.
    """
    db = load_users()
    u = form_data["username"]
    if not all(form_data.values()):
        raise ValueError("All fields are required.")
    if form_data["password"] != form_data["confirm_password"]:
        raise ValueError("Passwords must match.")
    if u in db["usernames"]:
        raise ValueError("Username already exists.")
    # Hash password
    pw_hash = hashlib.sha256(form_data["password"].encode()).hexdigest()
    db["usernames"][u] = {
        "name": form_data["name"],
        "email": form_data["email"],
        "password": pw_hash,
        "role": form_data["role"]
    }
    save_users(db)
    return True

def login_user(username: str, password: str) -> Dict[str, str]:
    """
    Authenticate a user.
    Returns user info dict on success, raises ValueError on failure.
    """
    db = load_users()
    if username not in db["usernames"]:
        raise ValueError("Unknown username.")
    rec = db["usernames"][username]
    if hashlib.sha256(password.encode()).hexdigest() != rec["password"]:
        raise ValueError("Invalid credentials.")
    return {
        "username": username,
        "name": rec["name"],
        "role": rec["role"],
        "email": rec["email"]
    }

def logout_user():
    """
    Placeholder for logout logic.
    In Streamlit app clear session_state keys manually.
    """
    pass

# ─── CONTEXT CHUNK CACHING ─────────────────────────────────────────────────────────

class ContextCache:
    """
    Two-layer context chunk cache:
      1. Exact-match (hash-based)
      2. Semantic-match (embedding similarity)
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        exact_cache_path: str = "exact_chunk_cache.pkl",
        sem_cache_path: str = "sem_chunk_cache.pkl",
        sem_threshold: float = 0.55  # LOWERED FROM 0.95 TO 0.75
    ):
        self.model = embedding_model
        self.exact_cache_path = exact_cache_path
        self.sem_cache_path = sem_cache_path
        self.sem_threshold = sem_threshold
        self._load_caches()

    def _load_caches(self):
        self.exact_cache: Dict[str, str] = {}
        self.sem_cache: List[Any] = []  # list of (embedding, chunk_text)
        if os.path.exists(self.exact_cache_path):
            with open(self.exact_cache_path, "rb") as f:
                self.exact_cache = pickle.load(f)
        if os.path.exists(self.sem_cache_path):
            with open(self.sem_cache_path, "rb") as f:
                self.sem_cache = pickle.load(f)

    def _save_exact(self):
        with open(self.exact_cache_path, "wb") as f:
            pickle.dump(self.exact_cache, f)

    def _save_sem(self):
        with open(self.sem_cache_path, "wb") as f:
            pickle.dump(self.sem_cache, f)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_chunk(self, chunk: str) -> str:
        """
        Return a cached chunk if available; otherwise add to cache.
        Logs hits and misses to console for debugging.
        """
        h = self._hash(chunk)
        # Exact-match cache
        if h in self.exact_cache:
            print(f"[ContextCache] exact hit for hash {h[:8]}")
            return self.exact_cache[h]
        # Semantic-match cache
        emb = self.model.encode([chunk], show_progress_bar=False)[0]
        for e, txt in self.sem_cache:
            sim = np.dot(e, emb) / (np.linalg.norm(e) * np.linalg.norm(emb))
            if sim >= self.sem_threshold:
                print(f"[ContextCache] semantic hit (sim={sim:.2f})")
                return txt
        # Cache miss
        print(f"[ContextCache] MISS for chunk hash {h[:8]}, adding to cache")
        self.exact_cache[h] = chunk
        self.sem_cache.append((emb, chunk))
        self._save_exact()
        self._save_sem()
        return chunk

    def clear_caches(self):
        """Clear both exact and semantic caches."""
        self.exact_cache.clear()
        self.sem_cache.clear()
        self._save_exact()
        self._save_sem()
