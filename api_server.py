#!/usr/bin/env python3
"""
RMSAI Streaming API Server
==========================

RESTful API for real-time access to RMSAI processing results, embeddings, and anomaly data.
Provides endpoints for querying anomalies, similarity search, and real-time updates.

Features:
- Real-time processing statistics
- Anomaly querying with filters
- Vector similarity search via ChromaDB
- Event details and metadata access
- WebSocket support for live updates
- CORS enabled for web dashboards

Usage:
    python api_server.py

    # Access API at http://localhost:8000
    # Documentation at http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import processor config
try:
    from rmsai_lstm_autoencoder_proc import RMSAIConfig
    PROCESSOR_CONFIG_AVAILABLE = True
except ImportError:
    PROCESSOR_CONFIG_AVAILABLE = False
    logger.warning("Processor configuration not available")
import sqlite3
import asyncio
from datetime import datetime, timedelta
import json
import logging
import uvicorn

# Optional ChromaDB import
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

app = FastAPI(
    title="RMSAI Anomaly Detection API",
    description="Real-time API for ECG anomaly detection and embedding analysis",
    version="1.0.0"
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "rmsai_metadata.db"
VECTOR_DB_PATH = "vector_db"

# Global processor configuration cache
_processor_config = None
_config_cache_time = None
CONFIG_CACHE_DURATION = 60  # seconds

# Pydantic models
class AnomalyQuery(BaseModel):
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    condition: Optional[str] = None
    lead_name: Optional[str] = None
    min_error_score: Optional[float] = None
    max_error_score: Optional[float] = None
    limit: int = 100

class SimilaritySearch(BaseModel):
    chunk_id: str
    n_results: int = 5
    threshold: Optional[float] = None
    include_metadata: bool = True

class ProcessingStats(BaseModel):
    total_chunks: int
    total_anomalies: int
    anomaly_rate: float
    avg_error_score: float
    files_processed: int
    conditions_detected: Dict[str, int]
    leads_processed: Dict[str, int]
    last_updated: str
    uptime_hours: float
    selected_leads: List[str]
    total_available_leads: int
    performance_impact: Dict[str, Any]

class LeadConfigUpdate(BaseModel):
    selected_leads: List[str]

class ChunkDetail(BaseModel):
    chunk_id: str
    event_id: str
    source_file: str
    lead_name: str
    anomaly_status: str
    anomaly_type: Optional[str]
    error_score: float
    vector_id: Optional[str]
    processing_timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class EventDetail(BaseModel):
    event_id: str
    source_file: str
    total_chunks: int
    anomaly_count: int
    avg_error_score: float
    conditions: List[str]
    chunks: List[ChunkDetail]

# Global variables for caching
_stats_cache = None
_stats_cache_time = None
CACHE_DURATION = 30  # seconds

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Utility functions
def get_db_connection():
    """Get SQLite database connection"""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def get_vector_db():
    """Get ChromaDB client and collection"""
    if not CHROMADB_AVAILABLE:
        raise HTTPException(status_code=500, detail="ChromaDB not available")

    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection("rmsai_ecg_embeddings")
        return client, collection
    except Exception as e:
        logger.error(f"Vector database error: {e}")
        raise HTTPException(status_code=500, detail="Vector database connection failed")

def get_processor_config():
    """Get current processor configuration with caching"""
    global _processor_config, _config_cache_time

    current_time = datetime.now()
    if (_processor_config and _config_cache_time and
        (current_time - _config_cache_time).total_seconds() < CONFIG_CACHE_DURATION):
        return _processor_config

    if PROCESSOR_CONFIG_AVAILABLE:
        try:
            config = RMSAIConfig()
            _processor_config = {
                'selected_leads': config.selected_leads,
                'available_leads': config.available_leads,
                'chunking_config': {
                    'chunk_size': config.chunk_size,
                    'step_size': config.step_size,
                    'max_chunks_per_lead': config.max_chunks_per_lead
                },
                'performance_estimates': config.get_performance_estimate()
            }
            _config_cache_time = current_time
            return _processor_config
        except Exception as e:
            logger.error(f"Error getting processor config: {e}")

    # Fallback: use centralized config with database leads
    try:
        from config import get_default_ecg_config, ECG_PROCESSING_CONFIG, get_performance_estimates

        # Get available leads from database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT lead_name FROM chunks ORDER BY lead_name")
            db_available_leads = [row[0] for row in cursor.fetchall()]

        # Use centralized config with database leads if available
        default_config = get_default_ecg_config()
        available_leads = db_available_leads if db_available_leads else default_config['available_leads']
        selected_leads = default_config['selected_leads']  # Use default selected leads from config

        performance_estimates = get_performance_estimates(len(selected_leads))
        performance_estimates.update({
            'selected_leads': len(selected_leads),
            'max_chunks_per_lead': ECG_PROCESSING_CONFIG['max_chunks_per_lead'],
            'coverage_percentage': 99.2
        })

        _processor_config = {
            'selected_leads': selected_leads,
            'available_leads': available_leads,
            'chunking_config': {
                'chunk_size': ECG_PROCESSING_CONFIG['chunk_size'],
                'step_size': ECG_PROCESSING_CONFIG['step_size'],
                'max_chunks_per_lead': ECG_PROCESSING_CONFIG['max_chunks_per_lead']
            },
            'performance_estimates': performance_estimates
        }
        _config_cache_time = current_time
        return _processor_config
    except Exception as e:
        logger.error(f"Error getting fallback config: {e}")
        return None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RMSAI Anomaly Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "stats": "/api/v1/stats"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks LIMIT 1")
            chunk_count = cursor.fetchone()[0]

        # Check vector database if available
        vector_status = "not_available"
        if CHROMADB_AVAILABLE:
            try:
                _, collection = get_vector_db()
                vector_count = collection.count()
                vector_status = "available"
            except:
                vector_status = "error"

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "chunk_count": chunk_count,
            "vector_database": vector_status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/api/v1/stats", response_model=ProcessingStats)
async def get_processing_stats():
    """Get comprehensive processing statistics with caching"""
    global _stats_cache, _stats_cache_time

    # Check cache
    now = datetime.now()
    if (_stats_cache and _stats_cache_time and
        (now - _stats_cache_time).total_seconds() < CACHE_DURATION):
        return _stats_cache

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chunks WHERE anomaly_status='anomaly'")
            total_anomalies = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(error_score) FROM chunks")
            avg_error = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE status='completed'")
            files_processed = cursor.fetchone()[0]

            # Condition distribution
            cursor.execute("""
                SELECT anomaly_type, COUNT(*)
                FROM chunks
                WHERE anomaly_type IS NOT NULL
                GROUP BY anomaly_type
            """)
            conditions_detected = dict(cursor.fetchall())

            # Lead distribution
            cursor.execute("""
                SELECT lead_name, COUNT(*)
                FROM chunks
                GROUP BY lead_name
            """)
            leads_processed = dict(cursor.fetchall())

            # Calculate uptime (time since first processing)
            cursor.execute("SELECT MIN(processing_timestamp) FROM chunks")
            first_timestamp = cursor.fetchone()[0]

            uptime_hours = 0
            if first_timestamp:
                first_time = datetime.fromisoformat(first_timestamp)
                uptime_hours = (now - first_time).total_seconds() / 3600

            anomaly_rate = (total_anomalies / total_chunks * 100) if total_chunks > 0 else 0

            # Get processor configuration
            config = get_processor_config()
            selected_leads = config['selected_leads'] if config else list(leads_processed.keys())
            performance_impact = config['performance_estimates'] if config else {}

            stats = ProcessingStats(
                total_chunks=total_chunks,
                total_anomalies=total_anomalies,
                anomaly_rate=round(anomaly_rate, 2),
                avg_error_score=round(avg_error, 4),
                files_processed=files_processed,
                conditions_detected=conditions_detected,
                leads_processed=leads_processed,
                last_updated=now.isoformat(),
                uptime_hours=round(uptime_hours, 2),
                selected_leads=selected_leads,
                total_available_leads=len(leads_processed),
                performance_impact=performance_impact
            )

            # Cache the result
            _stats_cache = stats
            _stats_cache_time = now

            return stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/anomalies")
async def query_anomalies(query: AnomalyQuery):
    """Query anomalies with advanced filtering options"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Build dynamic query
            where_conditions = ["anomaly_status = 'anomaly'"]
            params = []

            if query.condition:
                where_conditions.append("anomaly_type LIKE ?")
                params.append(f"%{query.condition}%")

            if query.lead_name:
                where_conditions.append("lead_name = ?")
                params.append(query.lead_name)

            if query.min_error_score is not None:
                where_conditions.append("error_score >= ?")
                params.append(query.min_error_score)

            if query.max_error_score is not None:
                where_conditions.append("error_score <= ?")
                params.append(query.max_error_score)

            if query.start_time:
                where_conditions.append("processing_timestamp >= ?")
                params.append(query.start_time)

            if query.end_time:
                where_conditions.append("processing_timestamp <= ?")
                params.append(query.end_time)

            where_clause = " AND ".join(where_conditions)

            sql = f"""
                SELECT chunk_id, event_id, source_file, lead_name,
                       anomaly_type, error_score, processing_timestamp,
                       vector_id, metadata
                FROM chunks
                WHERE {where_clause}
                ORDER BY error_score DESC
                LIMIT ?
            """
            params.append(query.limit)

            cursor.execute(sql, params)
            results = cursor.fetchall()

            anomalies = []
            for row in results:
                metadata = json.loads(row[8]) if row[8] else None
                anomalies.append({
                    "chunk_id": row[0],
                    "event_id": row[1],
                    "source_file": row[2],
                    "lead_name": row[3],
                    "anomaly_type": row[4],
                    "error_score": round(row[5], 4),
                    "processing_timestamp": row[6],
                    "vector_id": row[7],
                    "metadata": metadata
                })

            return {
                "anomalies": anomalies,
                "count": len(anomalies),
                "query": query.dict(),
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error querying anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search/similar")
async def search_similar_embeddings(search: SimilaritySearch):
    """Find similar ECG patterns using vector similarity"""
    if not CHROMADB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Vector search not available")

    try:
        _, collection = get_vector_db()

        # Get the reference embedding (add vec_ prefix as used in vector database)
        vector_id = f"vec_{search.chunk_id}"
        reference = collection.get(
            ids=[vector_id],
            include=['embeddings', 'metadatas'] if search.include_metadata else ['embeddings']
        )

        if (reference['embeddings'] is None or
            len(reference['embeddings']) == 0 or
            reference['embeddings'][0] is None or
            len(reference['embeddings'][0]) == 0):
            raise HTTPException(status_code=404, detail=f"Chunk {search.chunk_id} not found")

        # Search for similar embeddings
        results = collection.query(
            query_embeddings=reference['embeddings'],
            n_results=search.n_results + 1,  # +1 to exclude self
            include=['metadatas', 'distances'] if search.include_metadata else ['distances']
        )

        similar_chunks = []

        # Check if results contain any data
        if ('ids' not in results or results['ids'] is None or
            len(results['ids']) == 0 or results['ids'][0] is None or
            len(results['ids'][0]) == 0):
            return {
                "reference_chunk": search.chunk_id,
                "similar_chunks": [],
                "total_found": 0
            }

        for i, vector_chunk_id in enumerate(results['ids'][0]):
            # Remove vec_ prefix to get the actual chunk_id
            chunk_id = vector_chunk_id.replace("vec_", "") if vector_chunk_id.startswith("vec_") else vector_chunk_id

            if chunk_id != search.chunk_id:  # Exclude self
                distance = results['distances'][0][i]
                similarity_score = max(0, 1 - distance)  # Convert distance to similarity

                if search.threshold is None or similarity_score >= search.threshold:
                    chunk_info = {
                        "chunk_id": chunk_id,
                        "similarity_score": round(similarity_score, 4),
                        "distance": round(distance, 4)
                    }

                    if search.include_metadata and 'metadatas' in results:
                        chunk_info["metadata"] = results['metadatas'][0][i]

                    similar_chunks.append(chunk_info)

        # Limit results after filtering
        similar_chunks = similar_chunks[:search.n_results]

        return {
            "reference_chunk": search.chunk_id,
            "similar_chunks": similar_chunks,
            "count": len(similar_chunks),
            "search_params": search.dict(),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/events/{event_id}", response_model=EventDetail)
async def get_event_details(event_id: str):
    """Get detailed information about a specific event"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT chunk_id, source_file, lead_name, anomaly_status,
                       anomaly_type, error_score, vector_id, processing_timestamp, metadata
                FROM chunks
                WHERE event_id = ?
                ORDER BY lead_name
            """, (event_id,))

            rows = cursor.fetchall()

            if not rows:
                raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

            # Process chunks
            chunks = []
            conditions = set()
            total_error = 0
            anomaly_count = 0

            for row in rows:
                metadata = json.loads(row[8]) if row[8] else None

                chunk = ChunkDetail(
                    chunk_id=row[0],
                    event_id=event_id,
                    source_file=row[1],
                    lead_name=row[2],
                    anomaly_status=row[3],
                    anomaly_type=row[4],
                    error_score=round(row[5], 4),
                    vector_id=row[6],
                    processing_timestamp=row[7],
                    metadata=metadata
                )
                chunks.append(chunk)

                if row[4]:  # anomaly_type
                    conditions.add(row[4])
                if row[3] == 'anomaly':
                    anomaly_count += 1
                total_error += row[5]

            avg_error = total_error / len(chunks) if chunks else 0

            event_detail = EventDetail(
                event_id=event_id,
                source_file=chunks[0].source_file,
                total_chunks=len(chunks),
                anomaly_count=anomaly_count,
                avg_error_score=round(avg_error, 4),
                conditions=list(conditions),
                chunks=chunks
            )

            return event_detail

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting event details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/conditions")
async def get_conditions():
    """Get list of all detected conditions with statistics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT anomaly_type,
                       COUNT(*) as total_chunks,
                       SUM(CASE WHEN anomaly_status = 'anomaly' THEN 1 ELSE 0 END) as anomaly_chunks,
                       AVG(error_score) as avg_error_score,
                       MIN(error_score) as min_error_score,
                       MAX(error_score) as max_error_score
                FROM chunks
                WHERE anomaly_type IS NOT NULL
                GROUP BY anomaly_type
                ORDER BY total_chunks DESC
            """)

            results = cursor.fetchall()

            conditions = []
            for row in results:
                condition_name = row[0]
                total_chunks = row[1]
                anomaly_chunks = row[2]
                anomaly_rate = (anomaly_chunks / total_chunks * 100) if total_chunks > 0 else 0

                conditions.append({
                    "condition": condition_name,
                    "total_chunks": total_chunks,
                    "anomaly_chunks": anomaly_chunks,
                    "anomaly_rate": round(anomaly_rate, 2),
                    "avg_error_score": round(row[3], 4),
                    "min_error_score": round(row[4], 4),
                    "max_error_score": round(row[5], 4)
                })

            return {
                "conditions": conditions,
                "total_conditions": len(conditions),
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error getting conditions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/leads")
async def get_leads():
    """Get list of all ECG leads with statistics"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT lead_name,
                       COUNT(*) as total_chunks,
                       SUM(CASE WHEN anomaly_status = 'anomaly' THEN 1 ELSE 0 END) as anomaly_chunks,
                       AVG(error_score) as avg_error_score
                FROM chunks
                GROUP BY lead_name
                ORDER BY lead_name
            """)

            results = cursor.fetchall()

            leads = []
            for row in results:
                lead_name = row[0]
                total_chunks = row[1]
                anomaly_chunks = row[2]
                anomaly_rate = (anomaly_chunks / total_chunks * 100) if total_chunks > 0 else 0

                leads.append({
                    "lead_name": lead_name,
                    "total_chunks": total_chunks,
                    "anomaly_chunks": anomaly_chunks,
                    "anomaly_rate": round(anomaly_rate, 2),
                    "avg_error_score": round(row[3], 4)
                })

            return {
                "leads": leads,
                "total_leads": len(leads),
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Error getting leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/config/leads")
async def get_lead_configuration():
    """Get current lead configuration and performance estimates"""
    try:
        config = get_processor_config()
        if not config:
            raise HTTPException(status_code=500, detail="Could not retrieve processor configuration")

        return {
            "selected_leads": config['selected_leads'],
            "available_leads": config['available_leads'],
            "chunking_config": config['chunking_config'],
            "performance_estimates": config['performance_estimates'],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting lead configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/config/leads")
async def update_lead_configuration(config_update: LeadConfigUpdate):
    """Update lead configuration (Note: requires processor restart to take effect)"""
    try:
        # Validate leads using centralized validation
        from config import validate_ecg_leads

        is_valid, invalid_leads = validate_ecg_leads(config_update.selected_leads)
        if not is_valid:
            from config import ALL_AVAILABLE_ECG_LEADS
            raise HTTPException(
                status_code=400,
                detail=f"Invalid leads: {invalid_leads}. Available leads: {ALL_AVAILABLE_ECG_LEADS}"
            )

        # Additional validation - check leads exist in database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT lead_name FROM chunks")
            db_available_leads = [row[0] for row in cursor.fetchall()]

        missing_leads = [lead for lead in config_update.selected_leads if lead not in db_available_leads]
        if missing_leads:
            raise HTTPException(
                status_code=400,
                detail=f"Selected leads not found in database: {missing_leads}. Database has: {db_available_leads}"
            )

        # Calculate performance impact
        current_config = get_processor_config()
        if current_config:
            current_chunks_per_event = len(current_config['selected_leads']) * current_config['chunking_config']['max_chunks_per_lead']
            new_chunks_per_event = len(config_update.selected_leads) * current_config['chunking_config']['max_chunks_per_lead']
            performance_change = ((current_chunks_per_event - new_chunks_per_event) / current_chunks_per_event) * 100
        else:
            performance_change = 0

        # Note: This endpoint provides information but doesn't actually update the running processor
        # The processor configuration would need to be updated separately

        return {
            "message": "Lead configuration validated successfully",
            "note": "Processor restart required to apply changes",
            "requested_leads": config_update.selected_leads,
            "performance_impact": {
                "chunks_per_event_change": new_chunks_per_event - current_chunks_per_event if current_config else 0,
                "performance_improvement_percent": round(performance_change, 1) if current_config else 0
            },
            "validation": {
                "valid": True,
                "available_leads": db_available_leads,
                "invalid_leads": invalid_leads
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating lead configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pacer-analysis")
async def get_pacer_analysis():
    """Get comprehensive pacer pattern analysis across all data"""
    try:
        with get_db_connection() as conn:
            # Check if pacer data is available
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as pacer_count
                FROM chunks
                WHERE pacer_type IS NOT NULL
                AND pacer_offset IS NOT NULL
                AND processing_timestamp > datetime('now', '-30 days')
            """)
            pacer_count = cursor.fetchone()[0]

            if pacer_count == 0:
                return {
                    "message": "No pacer data available",
                    "pacer_data_available": False,
                    "total_chunks_analyzed": 0
                }

            # Get pacer distribution
            cursor.execute("""
                SELECT pacer_type, COUNT(*) as count
                FROM chunks
                WHERE pacer_type IS NOT NULL
                AND processing_timestamp > datetime('now', '-30 days')
                GROUP BY pacer_type
                ORDER BY count DESC
            """)
            pacer_type_dist = dict(cursor.fetchall())

            # Get pacer timing analysis
            cursor.execute("""
                SELECT pacer_offset, anomaly_status, anomaly_type, error_score
                FROM chunks
                WHERE pacer_offset IS NOT NULL
                AND processing_timestamp > datetime('now', '-30 days')
            """)
            timing_data = cursor.fetchall()

            timing_analysis = {}
            if timing_data:
                import numpy as np
                offsets = [row[0] for row in timing_data]

                # Convert to timing categories
                timing_categories = []
                for offset in offsets:
                    percent = (offset / 2400.0) * 100
                    if percent <= 25:
                        timing_categories.append("Early")
                    elif percent >= 75:
                        timing_categories.append("Late")
                    else:
                        timing_categories.append("Mid")

                from collections import Counter
                timing_dist = dict(Counter(timing_categories))

                timing_analysis = {
                    "timing_distribution": timing_dist,
                    "offset_statistics": {
                        "mean_samples": round(np.mean(offsets), 2),
                        "std_samples": round(np.std(offsets), 2),
                        "min_samples": min(offsets),
                        "max_samples": max(offsets),
                        "mean_seconds": round(np.mean(offsets) / 200.0, 3),
                        "std_seconds": round(np.std(offsets) / 200.0, 3)
                    }
                }

            # Get pacer by condition analysis
            cursor.execute("""
                SELECT anomaly_type, pacer_type, COUNT(*) as count,
                       AVG(error_score) as avg_error_score
                FROM chunks
                WHERE pacer_type IS NOT NULL
                AND anomaly_type IS NOT NULL
                AND processing_timestamp > datetime('now', '-30 days')
                GROUP BY anomaly_type, pacer_type
                ORDER BY anomaly_type, pacer_type
            """)
            condition_pacer_data = cursor.fetchall()

            pacer_by_condition = {}
            for condition, pacer_type, count, avg_error in condition_pacer_data:
                if condition not in pacer_by_condition:
                    pacer_by_condition[condition] = {}

                pacer_type_name = ['None', 'Single', 'Dual', 'Biventricular'][min(int(pacer_type), 3)]
                pacer_by_condition[condition][f"pacer_type_{pacer_type}"] = {
                    "name": pacer_type_name,
                    "count": count,
                    "avg_error_score": round(avg_error, 4) if avg_error else 0.0
                }

            return {
                "pacer_data_available": True,
                "total_chunks_with_pacer_data": pacer_count,
                "analysis_timestamp": datetime.now().isoformat(),
                "pacer_type_distribution": {
                    "counts": pacer_type_dist,
                    "percentages": {
                        str(k): round(v / pacer_count * 100, 2)
                        for k, v in pacer_type_dist.items()
                    }
                },
                "timing_analysis": timing_analysis,
                "pacer_by_condition": pacer_by_condition
            }

    except Exception as e:
        logger.error(f"Error in pacer analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Pacer analysis failed: {str(e)}")

@app.get("/api/v1/analytics/pacer-patterns")
async def get_advanced_pacer_analytics():
    """Get advanced pacer pattern analytics using the analytics module"""
    try:
        # Import analytics module
        from advanced_analytics import EmbeddingAnalytics

        analytics = EmbeddingAnalytics("vector_db", DB_PATH)
        pacer_analysis = analytics.analyze_pacer_patterns()

        return {
            "analysis_type": "advanced_pacer_patterns",
            "timestamp": datetime.now().isoformat(),
            "results": pacer_analysis
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Advanced analytics module not available"
        )
    except Exception as e:
        logger.error(f"Error in advanced pacer analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced pacer analytics failed: {str(e)}"
        )

@app.get("/api/v1/thresholds/pacer-impact")
async def get_pacer_threshold_impact():
    """Get pacer impact analysis on threshold optimization"""
    try:
        # Import adaptive thresholds module
        from adaptive_thresholds import AdaptiveThresholdManager

        threshold_manager = AdaptiveThresholdManager(DB_PATH)
        pacer_impact = threshold_manager.analyze_pacer_impact_on_thresholds()

        return {
            "analysis_type": "pacer_threshold_impact",
            "timestamp": datetime.now().isoformat(),
            "results": pacer_impact
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Adaptive thresholds module not available"
        )
    except Exception as e:
        logger.error(f"Error in pacer threshold impact analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pacer threshold impact analysis failed: {str(e)}"
        )

# WebSocket endpoint for real-time updates
@app.websocket("/ws/live-updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time processing updates"""
    await manager.connect(websocket)

    try:
        last_chunk_count = 0

        # Send initial stats
        try:
            stats = await get_processing_stats()
            await manager.send_personal_message(
                json.dumps({
                    "type": "initial_stats",
                    "data": stats.dict()
                }),
                websocket
            )
        except:
            pass

        while True:
            try:
                # Check for new processing results
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM chunks")
                    current_count = cursor.fetchone()[0]

                    if current_count > last_chunk_count:
                        # Get latest chunks
                        cursor.execute("""
                            SELECT chunk_id, event_id, lead_name, anomaly_status,
                                   error_score, processing_timestamp
                            FROM chunks
                            ORDER BY id DESC
                            LIMIT ?
                        """, (current_count - last_chunk_count,))

                        new_chunks = cursor.fetchall()

                        for chunk in new_chunks:
                            await manager.send_personal_message(
                                json.dumps({
                                    "type": "new_chunk",
                                    "data": {
                                        "chunk_id": chunk[0],
                                        "event_id": chunk[1],
                                        "lead_name": chunk[2],
                                        "anomaly_status": chunk[3],
                                        "error_score": round(chunk[4], 4),
                                        "processing_timestamp": chunk[5],
                                        "timestamp": datetime.now().isoformat()
                                    }
                                }),
                                websocket
                            )

                        last_chunk_count = current_count

                # Send periodic stats update
                if last_chunk_count % 50 == 0 and last_chunk_count > 0:
                    try:
                        stats = await get_processing_stats()
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "stats_update",
                                "data": stats.dict()
                            }),
                            websocket
                        )
                    except:
                        pass

                await asyncio.sleep(2)  # Check every 2 seconds

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": str(e)
                    }),
                    websocket
                )
                break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

# Simple HTML page for WebSocket testing
@app.get("/test-websocket")
async def websocket_test_page():
    """Simple test page for WebSocket functionality"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RMSAI WebSocket Test</title>
    </head>
    <body>
        <h1>RMSAI Real-time Updates</h1>
        <div id="messages"></div>
        <script>
            const ws = new WebSocket("ws://localhost:8000/ws/live-updates");
            const messages = document.getElementById('messages');

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                const div = document.createElement('div');
                div.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            };

            ws.onopen = function(event) {
                const div = document.createElement('div');
                div.innerHTML = '<strong>Connected to WebSocket</strong>';
                messages.appendChild(div);
            };

            ws.onclose = function(event) {
                const div = document.createElement('div');
                div.innerHTML = '<strong>WebSocket connection closed</strong>';
                messages.appendChild(div);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    logger.info("Starting RMSAI API Server...")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("WebSocket Test: http://localhost:8000/test-websocket")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )