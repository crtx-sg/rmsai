# RMSAI LSTM Autoencoder Processor

## Overview

The RMSAI LSTM Autoencoder Processor is a real-time processing pipeline for ECG anomaly detection using LSTM autoencoders. It monitors the data directory for new HDF5 files, processes ECG chunks through the autoencoder for anomaly detection and embedding generation.

## Features

### Core Functionality
- **Real-time File Monitoring**: Watches data directory for new `.h5` files using pyinotify (Linux) or polling (cross-platform)
- **Multi-lead ECG Processing**: Processes all 7 ECG leads (ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX)
- **LSTM Autoencoder Integration**: Uses pre-trained model for encoding/decoding ECG chunks
- **Anomaly Detection**: Configurable threshold-based anomaly detection with condition-specific thresholds
- **Vector Database Storage**: Stores embeddings in ChromaDB for similarity search
- **Metadata Storage**: SQLite database for processing metadata and results
- **Graceful Error Handling**: Robust error handling with detailed logging
- **Pre-trained Model Support**: Compatible with models from Colab training

### Processing Pipeline
1. **File Detection**: Monitor data directory for new HDF5 files
2. **File Loading**: Load and validate HDF5 structure
3. **Event Processing**: Extract ECG data from each event
4. **Chunk Processing**: Process each ECG lead (12-second chunks)
5. **Encoding**: Generate embeddings using LSTM encoder
6. **Decoding**: Reconstruct ECG from embeddings
7. **Anomaly Detection**: Compare reconstruction error to thresholds
8. **Storage**: Store embeddings and metadata in databases

## Architecture

### Components

#### RMSAIConfig
Configuration management for the processor:
- Directory paths and file locations
- Model configuration (sequence length, embedding dimensions)
- ECG leads configuration
- Anomaly detection thresholds
- Processing parameters

#### ModelManager
Handles LSTM autoencoder operations:
- Pre-trained model loading with multiple fallback strategies
- ECG chunk preprocessing and normalization
- Encoding to vector embeddings
- Decoding back to ECG signals
- Anomaly detection based on reconstruction error

#### VectorDatabase (ChromaDB)
Manages vector embeddings storage:
- Persistent storage of ECG embeddings
- Metadata association with embeddings
- Similarity search capabilities
- Scalable vector operations

#### MetadataDatabase (SQLite)
Manages processing metadata:
- Chunk processing results
- File processing status
- Anomaly detection results
- Processing statistics

#### ECGChunkProcessor
Core processing logic for individual ECG chunks:
- Encoding/decoding workflow
- Anomaly classification
- Database storage coordination
- Result aggregation

#### HDF5FileProcessor
Handles HDF5 file operations:
- File validation and integrity checking
- Event extraction and processing
- Multi-lead ECG data extraction
- Error handling and recovery

#### FileMonitor
Real-time file monitoring:
- pyinotify-based monitoring (Linux)
- Polling-based fallback (cross-platform)
- Queue-based processing
- Duplicate detection

## Installation

### Dependencies
```bash
pip install -r requirements_processor.txt
```

### Required packages:
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `torch>=1.9.0` - LSTM autoencoder model
- `h5py>=3.1.0` - HDF5 file processing
- `scikit-learn>=1.0.0` - Data preprocessing
- `pyinotify>=0.9.6` - File monitoring (Linux only)
- `chromadb>=0.4.0` - Vector database

## Usage

### Basic Usage
```bash
# Start the processor
python rmsai_lstm_autoencoder_proc.py
```

### Configuration
The processor uses the `RMSAIConfig` class for configuration. Key settings:

```python
# Directory paths
data_dir = "data"                    # HDF5 files directory
models_dir = "models"                # Model storage
vector_db_dir = "vector_db"          # ChromaDB storage
sqlite_db_path = "rmsai_metadata.db" # SQLite database

# Model configuration
model_path = "model.pth"             # Pre-trained model
seq_len = 2400                       # 12 seconds at 200Hz
embedding_dim = 64                   # Embedding dimension

# Anomaly thresholds (condition-specific)
condition_thresholds = {
    'Normal': 0.05,
    'Tachycardia': 0.08,
    'Bradycardia': 0.07,
    'Atrial Fibrillation (PTB-XL)': 0.12,
    'Ventricular Tachycardia (MIT-BIH)': 0.15
}
```

### Testing
```bash
# Test the processor components
python test_processor.py
```

## Data Flow

### Input: HDF5 Files
Expected structure (from RMSAI generator):
```
PatientID_YYYY-MM.h5
├── metadata/
├── event_1001/
│   ├── ecg/
│   │   ├── ECG1, ECG2, ECG3
│   │   ├── aVR, aVL, aVF
│   │   └── vVX
│   ├── ppg/
│   ├── vitals/
│   ├── timestamp
│   └── uuid
└── event_100N/
```

### Processing Flow
1. **File Detection**: New `.h5` file created in data directory
2. **File Validation**: Check file integrity and structure
3. **Event Extraction**: Extract all events from HDF5 file
4. **Lead Processing**: Process each ECG lead (7 leads per event)

### Chunk Processing Workflow
For each ECG chunk (chunk_N1 through chunk_N7):

#### 1. Encoding
```python
chunk_data = ecg_lead[:]  # 2400 samples (12s @ 200Hz)
embedding = encoder(chunk_data)  # 64-dimensional vector
```

#### 2. Decoding
```python
reconstructed = decoder(embedding)  # Reconstructed ECG
```

#### 3. Anomaly Detection
```python
mse = mean_squared_error(original, reconstructed)
is_anomaly = mse > threshold
```

#### 4. Storage
- **Vector DB**: Store embedding with metadata
- **SQL DB**: Store processing results and anomaly status

### Output: Databases

#### ChromaDB (Vector Database)
```python
{
    "id": "chunk_10011",
    "embedding": [0.1, 0.2, ...],  # 64-dimensional vector
    "metadata": {
        "event_id": "event_1001",
        "source_file": "PT1234_2025-09.h5",
        "lead_name": "ECG1",
        "condition": "Ventricular Tachycardia",
        "error_score": 0.156,
        "anomaly_status": "anomaly"
    }
}
```

#### SQLite Database
**chunks table**:
| event_id | source_file | chunk_id | lead_name | vector_id | anomaly_status | anomaly_type | error_score |
|----------|-------------|----------|-----------|-----------|----------------|--------------|-------------|
| event_1001 | PT1234_2025-09.h5 | chunk_10011 | ECG1 | vec_chunk_10011 | anomaly | Ventricular Tachycardia | 0.156 |

## Anomaly Detection

### Threshold Configuration
The processor uses condition-specific thresholds for improved accuracy:

- **Normal**: 0.05 (tight threshold for normal ECGs)
- **Tachycardia**: 0.08 (moderate threshold)
- **Bradycardia**: 0.07 (moderate threshold)
- **Atrial Fibrillation**: 0.12 (higher threshold for irregular patterns)
- **Ventricular Tachycardia**: 0.15 (highest threshold for wide QRS)

### Error Metrics
- **MSE (Mean Squared Error)**: Primary metric for reconstruction quality
- **Condition-Aware**: Different thresholds based on clinical condition
- **Lead-Specific**: Each ECG lead processed independently

## Model Compatibility

### Pre-trained Model Loading
The processor supports multiple model loading strategies:

1. **Direct Loading**: Load complete model object
2. **State Dict Loading**: Load model weights with architecture matching
3. **Parameter Transfer**: Manual parameter copying with size matching
4. **Fallback**: Create new model if loading fails

### Model Requirements
- **Input**: 2400 samples (12 seconds at 200Hz)
- **Architecture**: LSTM Autoencoder (Encoder + Decoder)
- **Output**: 64-dimensional embedding vector
- **Framework**: PyTorch

## Performance

### Processing Speed
- **Single Chunk**: ~100ms per ECG chunk
- **Complete Event**: ~700ms (7 leads)
- **Throughput**: ~10 events per minute
- **Memory Usage**: <500MB during processing

### Storage Efficiency
- **HDF5 Input**: ~75KB per event (compressed)
- **Embedding Storage**: 256 bytes per chunk (64 float32 values)
- **Metadata**: ~200 bytes per chunk record
- **Compression Ratio**: ~300:1 for embeddings vs raw ECG

## Monitoring and Logging

### Log Levels
- **INFO**: Normal processing events, file detection, completion
- **WARNING**: Non-critical issues, fallback operations
- **ERROR**: Processing failures, database errors
- **DEBUG**: Detailed processing information

### Log Files
- **Console Output**: Real-time processing status
- **rmsai_processor.log**: Complete processing log with timestamps

### Statistics
Access processing statistics:
```python
stats = processor.get_processing_stats()
# Returns: total_chunks, anomaly_counts, files_processed, avg_error_score
```

## Error Handling

### File Processing Errors
- **Corrupt Files**: Skip and log error
- **Missing Data**: Handle gracefully with warnings
- **Permission Issues**: Retry with backoff
- **Duplicate Processing**: Check file hash to prevent reprocessing

### Model Errors
- **Loading Failures**: Multiple fallback strategies
- **Inference Errors**: Log and continue with next chunk
- **Memory Issues**: Batch processing with smaller chunks

### Database Errors
- **Connection Issues**: Retry with exponential backoff
- **Schema Errors**: Auto-create missing tables
- **Disk Space**: Monitor and alert on storage issues

## Integration

### Real-time Processing
The processor integrates with the RMSAI HDF5 generator:

1. **Generator**: Creates new HDF5 files in data directory
2. **Processor**: Detects and processes files automatically
3. **Storage**: Results available in vector and SQL databases
4. **Analysis**: Query databases for anomaly patterns

### API Integration
Extend the processor for API access:
```python
# Example REST API endpoints
GET /api/stats                    # Processing statistics
GET /api/anomalies               # Recent anomalies
POST /api/search                 # Vector similarity search
GET /api/events/{event_id}       # Event details
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```
WARNING - Creating new model (pre-trained model could not be loaded)
```
**Solution**: Ensure model.pth is compatible with current architecture

#### 2. pyinotify Not Available
```
Warning: pyinotify not available
```
**Solution**: Install pyinotify (`pip install pyinotify`) or use polling mode

#### 3. ChromaDB Errors
```
Error initializing ChromaDB
```
**Solution**: Check directory permissions and disk space

#### 4. High Memory Usage
**Solution**: Reduce batch size or processing threads in configuration

### Performance Tuning

#### 1. Increase Processing Speed
- Adjust `processing_threads` in configuration
- Use GPU if available (`device = 'cuda'`)
- Optimize batch processing

#### 2. Reduce Memory Usage
- Lower `max_queue_size`
- Process files sequentially
- Clear model cache periodically

#### 3. Storage Optimization
- Compress embeddings before storage
- Implement database cleanup routines
- Archive old results

## Future Enhancements

### Planned Features
1. **GPU Acceleration**: CUDA support for faster processing
2. **Distributed Processing**: Multi-node processing capability
3. **Real-time Alerts**: Anomaly notification system
4. **Web Dashboard**: Real-time monitoring interface
5. **Model Updates**: Dynamic model reloading
6. **Batch Processing**: High-throughput batch mode
7. **Advanced Analytics**: Pattern recognition and trending

### Extensibility
The processor is designed for easy extension:
- **Custom Models**: Support for different autoencoder architectures
- **Additional Databases**: Integration with other vector databases
- **Custom Anomaly Detection**: Pluggable anomaly detection algorithms
- **External APIs**: REST/GraphQL API integration
- **Streaming**: Real-time streaming data support

## License

This processor is part of the RMSAI project and follows the same licensing terms.

## Support

For issues and feature requests, refer to the RMSAI project documentation and issue tracker.