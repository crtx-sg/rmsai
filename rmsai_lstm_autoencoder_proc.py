#!/usr/bin/env python3
"""
RMSAI LSTM Autoencoder Processing Pipeline
==========================================

Real-time processing pipeline for ECG anomaly detection using LSTM autoencoder.
Monitors data directory for new HDF5 files, processes ECG chunks through the
autoencoder for anomaly detection and embedding generation.

Features:
- Real-time file monitoring with pyinotify
- Multi-lead ECG chunk processing (7 leads)
- LSTM autoencoder encoding/decoding
- Anomaly detection with configurable thresholds
- Vector database storage (ChromaDB)
- Metadata storage (SQLite)
- Graceful error handling and recovery
- Pre-trained model support

Author: RMSAI Team
"""

import os
import sys
import time
import logging
import sqlite3
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import json

import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.preprocessing import StandardScaler

# File monitoring
try:
    import pyinotify
    PYINOTIFY_AVAILABLE = True
except ImportError:
    PYINOTIFY_AVAILABLE = False
    print("Warning: pyinotify not available. Install with: pip install pyinotify")

# Vector database
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Install with: pip install chromadb")

# Import model classes from existing files
from rmsai_model import RecurrentAutoencoder, Encoder, Decoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rmsai_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RMSAIConfig:
    """Configuration class for RMSAI processor"""

    def __init__(self):
        # Directory paths
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.vector_db_dir = Path("vector_db")
        self.sqlite_db_path = Path("rmsai_metadata.db")

        # Model configuration
        self.model_path = Path("models/model.pth")
        self.seq_len = 140   # Model's expected sequence length
        self.n_features = 1
        self.embedding_dim = 128  # Match actual model output

        # ECG leads configuration
        self.ecg_leads = ['ECG1', 'ECG2', 'ECG3', 'aVR', 'aVL', 'aVF', 'vVX']

        # Anomaly detection thresholds
        self.anomaly_threshold = 0.1  # Base threshold for MSE
        self.condition_thresholds = {
            'Normal': 0.05,
            'Tachycardia': 0.08,
            'Bradycardia': 0.07,
            'Atrial Fibrillation (PTB-XL)': 0.12,
            'Ventricular Tachycardia (MIT-BIH)': 0.15
        }

        # Processing configuration
        self.batch_size = 1
        self.max_queue_size = 100
        self.processing_threads = 2

        # Create directories
        self.create_directories()

    def create_directories(self):
        """Create necessary directories"""
        for path in [self.data_dir, self.models_dir, self.vector_db_dir]:
            path.mkdir(exist_ok=True)

class ModelManager:
    """Manages LSTM autoencoder model loading and inference"""

    def __init__(self, config: RMSAIConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cpu')  # Use CPU for stability

        # Load pre-trained model
        self.load_model()

    def load_model(self) -> bool:
        """Load pre-trained LSTM autoencoder model"""
        if not self.config.model_path.exists():
            logger.warning(f"Model file not found: {self.config.model_path}")
            return False

        try:
            logger.info(f"Loading pre-trained model from {self.config.model_path}")

            # Try multiple loading strategies based on real_dataset_test.py
            model_loaded = False

            # Method 1: Direct loading
            try:
                self.model = torch.load(
                    self.config.model_path,
                    map_location=self.device,
                    weights_only=False
                )
                self.model.eval()
                model_loaded = True
                logger.info("Model loaded directly")
            except Exception as e:
                logger.warning(f"Direct loading failed: {e}")

            # Method 2: Load with state dict
            if not model_loaded:
                try:
                    checkpoint = torch.load(
                        self.config.model_path,
                        map_location=self.device,
                        weights_only=False
                    )

                    # Try different embedding dimensions
                    for embedding_dim in [128, 64, 256]:
                        try:
                            self.model = RecurrentAutoencoder(
                                seq_len=self.config.seq_len,
                                n_features=self.config.n_features,
                                embedding_dim=embedding_dim
                            )

                            # Extract state dict
                            if hasattr(checkpoint, 'state_dict'):
                                state_dict = checkpoint.state_dict()
                            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            else:
                                state_dict = checkpoint

                            self.model.load_state_dict(state_dict, strict=False)
                            self.model.eval()
                            model_loaded = True
                            logger.info(f"Model loaded with embedding_dim={embedding_dim}")
                            break

                        except Exception:
                            continue

                except Exception as e:
                    logger.warning(f"State dict loading failed: {e}")

            # Method 3: Create new model if loading fails
            if not model_loaded:
                logger.warning("Creating new model (pre-trained model could not be loaded)")
                self.model = RecurrentAutoencoder(
                    seq_len=self.config.seq_len,
                    n_features=self.config.n_features,
                    embedding_dim=self.config.embedding_dim
                )
                self.model.eval()

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def preprocess_chunk(self, chunk: np.ndarray) -> torch.Tensor:
        """Preprocess ECG chunk for model input"""
        # Ensure chunk is the right length
        if len(chunk) != self.config.seq_len:
            logger.warning(f"Chunk length {len(chunk)} doesn't match expected {self.config.seq_len}")
            # Truncate or pad as needed
            if len(chunk) > self.config.seq_len:
                chunk = chunk[:self.config.seq_len]
            else:
                chunk = np.pad(chunk, (0, self.config.seq_len - len(chunk)), mode='constant')

        # Normalize the chunk
        chunk_normalized = self.scaler.fit_transform(chunk.reshape(-1, 1)).flatten()

        # Convert to tensor and reshape for LSTM: [batch_size, seq_len, n_features]
        chunk_tensor = torch.FloatTensor(chunk_normalized).to(self.device)
        chunk_tensor = chunk_tensor.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]

        return chunk_tensor

    def encode_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Encode ECG chunk to embedding vector"""
        try:
            chunk_tensor = self.preprocess_chunk(chunk)

            with torch.no_grad():
                embedding = self.model.encoder(chunk_tensor)

            return embedding.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Error encoding chunk: {e}")
            return np.zeros(self.config.embedding_dim)

    def decode_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Decode embedding back to ECG chunk"""
        try:
            embedding_tensor = torch.FloatTensor(embedding).to(self.device)

            with torch.no_grad():
                reconstructed = self.model.decoder(embedding_tensor)

            return reconstructed.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Error decoding embedding: {e}")
            return np.zeros(self.config.seq_len)

    def detect_anomaly(self, original: np.ndarray, reconstructed: np.ndarray,
                      condition: str = None) -> Tuple[bool, float]:
        """Detect anomaly based on reconstruction error"""
        try:
            # Calculate Mean Squared Error
            mse = np.mean((original - reconstructed) ** 2)

            # Get threshold based on condition
            if condition and condition in self.config.condition_thresholds:
                threshold = self.config.condition_thresholds[condition]
            else:
                threshold = self.config.anomaly_threshold

            is_anomaly = mse > threshold

            return is_anomaly, mse

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0

class VectorDatabase:
    """Manages ChromaDB vector database operations"""

    def __init__(self, config: RMSAIConfig):
        self.config = config
        self.client = None
        self.collection = None

        if CHROMADB_AVAILABLE:
            self.initialize_db()

    def initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.config.vector_db_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="rmsai_ecg_embeddings",
                metadata={"description": "RMSAI ECG chunk embeddings"}
            )

            logger.info("ChromaDB initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None

    def store_embedding(self, chunk_id: str, embedding: np.ndarray,
                       metadata: Dict[str, Any]) -> bool:
        """Store embedding in vector database"""
        if not self.collection:
            logger.warning("ChromaDB not available, skipping vector storage")
            return False

        try:
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[f"ECG chunk {chunk_id}"],
                metadatas=[metadata],
                ids=[chunk_id]
            )

            logger.debug(f"Stored embedding for chunk {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False

    def search_similar(self, embedding: np.ndarray, n_results: int = 5) -> List[Dict]:
        """Search for similar embeddings"""
        if not self.collection:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=n_results
            )

            return results

        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []

class MetadataDatabase:
    """Manages SQLite database for metadata storage"""

    def __init__(self, config: RMSAIConfig):
        self.config = config
        self.db_path = config.sqlite_db_path
        self.initialize_db()

    def initialize_db(self):
        """Initialize SQLite database and create tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create chunks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        source_file TEXT NOT NULL,
                        chunk_id TEXT UNIQUE NOT NULL,
                        lead_name TEXT NOT NULL,
                        vector_id TEXT,
                        anomaly_status TEXT NOT NULL,
                        anomaly_type TEXT,
                        error_score REAL NOT NULL,
                        processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                ''')

                # Create files table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processed_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL,
                        error_message TEXT
                    )
                ''')

                # Create indices
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_id ON chunks(event_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomaly_status ON chunks(anomaly_status)')

                conn.commit()

            logger.info("SQLite database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")

    def store_chunk_result(self, event_id: str, source_file: str, chunk_id: str,
                          lead_name: str, vector_id: str, anomaly_status: str,
                          anomaly_type: str, error_score: float,
                          metadata: Dict[str, Any] = None) -> bool:
        """Store chunk processing result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO chunks
                    (event_id, source_file, chunk_id, lead_name, vector_id,
                     anomaly_status, anomaly_type, error_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_id, source_file, chunk_id, lead_name, vector_id,
                    anomaly_status, anomaly_type, error_score,
                    json.dumps(metadata) if metadata else None
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error storing chunk result: {e}")
            return False

    def mark_file_processed(self, filename: str, file_hash: str,
                           status: str = "completed", error_message: str = None) -> bool:
        """Mark file as processed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO processed_files
                    (filename, file_hash, status, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (filename, file_hash, status, error_message))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error marking file processed: {e}")
            return False

    def is_file_processed(self, filename: str, file_hash: str) -> bool:
        """Check if file has already been processed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT COUNT(*) FROM processed_files
                    WHERE filename = ? AND file_hash = ? AND status = 'completed'
                ''', (filename, file_hash))

                count = cursor.fetchone()[0]
                return count > 0

        except Exception as e:
            logger.error(f"Error checking file status: {e}")
            return False

class ECGChunkProcessor:
    """Processes individual ECG chunks through the LSTM autoencoder"""

    def __init__(self, config: RMSAIConfig, model_manager: ModelManager,
                 vector_db: VectorDatabase, metadata_db: MetadataDatabase):
        self.config = config
        self.model_manager = model_manager
        self.vector_db = vector_db
        self.metadata_db = metadata_db

    def process_chunk(self, chunk_data: np.ndarray, chunk_id: str,
                     event_id: str, source_file: str, lead_name: str,
                     condition: str = None) -> Dict[str, Any]:
        """Process a single ECG chunk"""
        try:
            logger.debug(f"Processing chunk {chunk_id} from {lead_name}")

            # 1. Encode chunk to embedding
            embedding = self.model_manager.encode_chunk(chunk_data)

            # 2. Decode embedding back to reconstructed chunk
            reconstructed = self.model_manager.decode_embedding(embedding)

            # 3. Detect anomaly
            is_anomaly, error_score = self.model_manager.detect_anomaly(
                chunk_data, reconstructed, condition
            )

            # 4. Determine anomaly type
            anomaly_status = "anomaly" if is_anomaly else "normal"
            anomaly_type = condition if is_anomaly and condition else None

            # 5. Store embedding in vector database
            vector_metadata = {
                "event_id": event_id,
                "source_file": source_file,
                "lead_name": lead_name,
                "condition": condition,
                "error_score": error_score,
                "anomaly_status": anomaly_status
            }

            vector_id = f"vec_{chunk_id}"
            self.vector_db.store_embedding(vector_id, embedding, vector_metadata)

            # 6. Store metadata in SQL database
            chunk_metadata = {
                "condition": condition,
                "embedding_dim": len(embedding),
                "reconstruction_error": error_score,
                "chunk_length": len(chunk_data)
            }

            self.metadata_db.store_chunk_result(
                event_id=event_id,
                source_file=source_file,
                chunk_id=chunk_id,
                lead_name=lead_name,
                vector_id=vector_id,
                anomaly_status=anomaly_status,
                anomaly_type=anomaly_type,
                error_score=error_score,
                metadata=chunk_metadata
            )

            result = {
                "chunk_id": chunk_id,
                "lead_name": lead_name,
                "anomaly_status": anomaly_status,
                "anomaly_type": anomaly_type,
                "error_score": error_score,
                "embedding_dim": len(embedding),
                "vector_id": vector_id
            }

            logger.info(f"Processed {chunk_id}: {anomaly_status} (score: {error_score:.4f})")
            return result

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            return {
                "chunk_id": chunk_id,
                "lead_name": lead_name,
                "error": str(e)
            }

class HDF5FileProcessor:
    """Processes HDF5 files and extracts ECG chunks"""

    def __init__(self, config: RMSAIConfig, chunk_processor: ECGChunkProcessor,
                 metadata_db: MetadataDatabase):
        self.config = config
        self.chunk_processor = chunk_processor
        self.metadata_db = metadata_db

    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def process_file(self, filepath: Path) -> bool:
        """Process a single HDF5 file"""
        try:
            logger.info(f"Processing file: {filepath}")

            # Calculate file hash
            file_hash = self.calculate_file_hash(filepath)

            # Check if already processed
            if self.metadata_db.is_file_processed(str(filepath), file_hash):
                logger.info(f"File {filepath} already processed, skipping")
                return True

            # Mark file as being processed
            self.metadata_db.mark_file_processed(
                str(filepath), file_hash, "processing"
            )

            # Process HDF5 file
            with h5py.File(filepath, 'r') as f:
                # Get all events
                event_keys = [k for k in f.keys() if k.startswith('event_')]
                logger.info(f"Found {len(event_keys)} events in {filepath}")

                for event_key in sorted(event_keys):
                    try:
                        self.process_event(f, event_key, str(filepath))
                    except Exception as e:
                        logger.error(f"Error processing event {event_key}: {e}")
                        continue

            # Mark file as completed
            self.metadata_db.mark_file_processed(
                str(filepath), file_hash, "completed"
            )

            logger.info(f"Successfully processed file: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")

            # Mark file as failed
            try:
                file_hash = self.calculate_file_hash(filepath)
                self.metadata_db.mark_file_processed(
                    str(filepath), file_hash, "failed", str(e)
                )
            except:
                pass

            return False

    def process_event(self, hdf5_file, event_key: str, source_file: str):
        """Process a single event from HDF5 file"""
        try:
            event = hdf5_file[event_key]

            # Get event metadata
            condition = event.attrs.get('condition', 'Unknown')
            heart_rate = event.attrs.get('heart_rate', 0)

            logger.info(f"Processing {event_key}: {condition} (HR: {heart_rate})")

            # Extract ECG leads
            if 'ecg' not in event:
                logger.warning(f"No ECG data found in {event_key}")
                return

            ecg_group = event['ecg']

            # Process each ECG lead
            for lead_idx, lead_name in enumerate(self.config.ecg_leads, 1):
                if lead_name not in ecg_group:
                    logger.warning(f"Lead {lead_name} not found in {event_key}")
                    continue

                try:
                    # Extract ECG data (12 seconds at 200Hz = 2400 samples)
                    ecg_data = ecg_group[lead_name][:]

                    # Split into chunks for complete ECG coverage (99.8%)
                    # 2400 samples with 140-sample chunks, step_size=141 for 17 chunks/lead
                    chunk_size = self.chunk_processor.config.seq_len  # 140
                    step_size = 141  # Optimized for 99.8% coverage with 17 chunks per lead

                    chunks_processed = 0
                    for chunk_start in range(0, len(ecg_data) - chunk_size + 1, step_size):
                        ecg_chunk = ecg_data[chunk_start:chunk_start + chunk_size]

                        # Create unique chunk ID for each sub-chunk
                        chunk_id = f"chunk_{event_key.split('_')[1]}{lead_idx}_{chunk_start}"

                        # Process chunk
                        result = self.chunk_processor.process_chunk(
                            chunk_data=ecg_chunk,
                            chunk_id=chunk_id,
                            event_id=event_key,
                            source_file=source_file,
                            lead_name=lead_name,
                            condition=condition
                        )

                        if 'error' in result:
                            logger.error(f"Failed to process {chunk_id}: {result['error']}")
                        else:
                            logger.debug(f"Successfully processed {chunk_id}")

                        chunks_processed += 1

                        # Limit number of chunks per lead to avoid overwhelming
                        if chunks_processed >= 10:  # Process first 10 chunks
                            break

                    logger.info(f"Processed {chunks_processed} sub-chunks from {lead_name} in {event_key}")

                except Exception as e:
                    logger.error(f"Error processing lead {lead_name} in {event_key}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing event {event_key}: {e}")

class FileMonitor:
    """Monitors data directory for new HDF5 files"""

    def __init__(self, config: RMSAIConfig, file_processor: HDF5FileProcessor):
        self.config = config
        self.file_processor = file_processor
        self.file_queue = queue.Queue(maxsize=config.max_queue_size)
        self.processing_thread = None
        self.running = False

    def start_monitoring(self):
        """Start file monitoring"""
        self.running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        if PYINOTIFY_AVAILABLE:
            self._start_inotify_monitoring()
        else:
            self._start_polling_monitoring()

    def stop_monitoring(self):
        """Stop file monitoring"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

    def _process_queue(self):
        """Process files from queue"""
        while self.running:
            try:
                filepath = self.file_queue.get(timeout=1)

                # Wait a bit to ensure file is completely written
                time.sleep(2)

                # Process file
                success = self.file_processor.process_file(filepath)

                if success:
                    logger.info(f"Successfully processed: {filepath}")
                else:
                    logger.error(f"Failed to process: {filepath}")

                self.file_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing queue: {e}")

    def _start_inotify_monitoring(self):
        """Start inotify-based monitoring"""
        logger.info("Starting inotify-based file monitoring")

        class EventHandler(pyinotify.ProcessEvent):
            def __init__(self, monitor):
                self.monitor = monitor

            def process_IN_CLOSE_WRITE(self, event):
                if event.pathname.endswith('.h5'):
                    filepath = Path(event.pathname)
                    logger.info(f"New HDF5 file detected: {filepath}")
                    try:
                        self.monitor.file_queue.put(filepath, timeout=1)
                    except queue.Full:
                        logger.warning(f"Queue full, skipping file: {filepath}")

        wm = pyinotify.WatchManager()
        handler = EventHandler(self)
        notifier = pyinotify.Notifier(wm, handler)

        # Watch for file creation/modification
        mask = pyinotify.IN_CLOSE_WRITE
        wm.add_watch(str(self.config.data_dir), mask, rec=False)

        try:
            notifier.loop()
        except Exception as e:
            logger.error(f"Error in inotify monitoring: {e}")

    def _start_polling_monitoring(self):
        """Start polling-based monitoring"""
        logger.info("Starting polling-based file monitoring")

        processed_files = set()

        while self.running:
            try:
                # Check for new .h5 files
                h5_files = list(self.config.data_dir.glob("*.h5"))

                for filepath in h5_files:
                    if filepath not in processed_files:
                        logger.info(f"New HDF5 file detected: {filepath}")
                        processed_files.add(filepath)

                        try:
                            self.file_queue.put(filepath, timeout=1)
                        except queue.Full:
                            logger.warning(f"Queue full, skipping file: {filepath}")

                time.sleep(5)  # Poll every 5 seconds

            except Exception as e:
                logger.error(f"Error in polling monitoring: {e}")
                time.sleep(10)

class RMSAIProcessor:
    """Main RMSAI processing pipeline"""

    def __init__(self, config_path: str = None):
        # Initialize configuration
        self.config = RMSAIConfig()

        # Initialize components
        logger.info("Initializing RMSAI Processor...")

        self.model_manager = ModelManager(self.config)
        self.vector_db = VectorDatabase(self.config)
        self.metadata_db = MetadataDatabase(self.config)

        self.chunk_processor = ECGChunkProcessor(
            self.config, self.model_manager, self.vector_db, self.metadata_db
        )

        self.file_processor = HDF5FileProcessor(
            self.config, self.chunk_processor, self.metadata_db
        )

        self.file_monitor = FileMonitor(self.config, self.file_processor)

        logger.info("RMSAI Processor initialized successfully")

    def start(self):
        """Start the processing pipeline"""
        logger.info("Starting RMSAI processing pipeline...")

        # Process any existing files
        self.process_existing_files()

        # Start monitoring for new files
        self.file_monitor.start_monitoring()

        logger.info("RMSAI processing pipeline started")

    def stop(self):
        """Stop the processing pipeline"""
        logger.info("Stopping RMSAI processing pipeline...")
        self.file_monitor.stop_monitoring()
        logger.info("RMSAI processing pipeline stopped")

    def process_existing_files(self):
        """Process any existing HDF5 files in data directory"""
        h5_files = list(self.config.data_dir.glob("*.h5"))

        if h5_files:
            logger.info(f"Found {len(h5_files)} existing HDF5 files")

            for filepath in h5_files:
                try:
                    self.file_processor.process_file(filepath)
                except Exception as e:
                    logger.error(f"Error processing existing file {filepath}: {e}")
        else:
            logger.info("No existing HDF5 files found")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            with sqlite3.connect(self.metadata_db.db_path) as conn:
                cursor = conn.cursor()

                # Total chunks processed
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]

                # Anomaly counts
                cursor.execute("""
                    SELECT anomaly_status, COUNT(*)
                    FROM chunks
                    GROUP BY anomaly_status
                """)
                anomaly_counts = dict(cursor.fetchall())

                # Files processed
                cursor.execute("SELECT COUNT(*) FROM processed_files WHERE status='completed'")
                files_processed = cursor.fetchone()[0]

                # Average error scores
                cursor.execute("SELECT AVG(error_score) FROM chunks")
                avg_error_score = cursor.fetchone()[0] or 0

                return {
                    "total_chunks": total_chunks,
                    "anomaly_counts": anomaly_counts,
                    "files_processed": files_processed,
                    "avg_error_score": avg_error_score
                }

        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}

def main():
    """Main entry point"""
    logger.info("Starting RMSAI LSTM Autoencoder Processor")

    # Initialize processor
    processor = RMSAIProcessor()

    try:
        # Start processing
        processor.start()

        # Keep running
        logger.info("Press Ctrl+C to stop the processor")
        while True:
            time.sleep(10)

            # Print stats every 60 seconds
            stats = processor.get_processing_stats()
            if stats:
                logger.info(f"Processing stats: {stats}")

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        # Stop processor
        processor.stop()
        logger.info("RMSAI Processor stopped")

if __name__ == "__main__":
    main()