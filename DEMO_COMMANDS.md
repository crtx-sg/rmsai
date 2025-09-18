# RMSAI Command Line Demo Guide
Complete demonstration of RMSAI system using command-line tools

---

## 1. System Status & Database Overview

### Check Database Tables and Records
```bash
# View database schema
sqlite3 rmsai_metadata.db ".schema"

# Count total records
sqlite3 rmsai_metadata.db "SELECT 'Chunks: ' || COUNT(*) FROM chunks; SELECT 'Files: ' || COUNT(*) FROM processed_files;"

# Show processing statistics
sqlite3 rmsai_metadata.db "
SELECT
    'Total Chunks: ' || COUNT(*) as stat FROM chunks
UNION ALL
SELECT 'Anomalies: ' || COUNT(*) FROM chunks WHERE anomaly_status = 'anomaly'
UNION ALL
SELECT 'Average Error Score: ' || ROUND(AVG(error_score), 4) FROM chunks
UNION ALL
SELECT 'Processing Time Span: ' ||
    (julianday(MAX(processing_timestamp)) - julianday(MIN(processing_timestamp))) * 24 || ' hours'
FROM chunks;
"
```

### Condition Distribution Analysis
```bash
sqlite3 rmsai_metadata.db "
SELECT
    'Condition Analysis:' as header, '' as count
UNION ALL
SELECT
    COALESCE(JSON_EXTRACT(metadata, '$.condition'), 'Unknown') as condition,
    COUNT(*) as count
FROM chunks
GROUP BY JSON_EXTRACT(metadata, '$.condition')
ORDER BY count DESC;
"
```

### ECG Lead Analysis
```bash
sqlite3 rmsai_metadata.db "
SELECT
    'ECG Lead Distribution:' as header, '' as count
UNION ALL
SELECT
    lead_name,
    COUNT(*) as count
FROM chunks
GROUP BY lead_name
ORDER BY count DESC;
"
```

---

## 2. Processing Pipeline Demonstrations

### Run Complete Analytics Pipeline
```bash
# Run advanced analytics (clustering, anomaly detection, temporal analysis)
python advanced_analytics.py
```

### Process New ECG File (if available)
```bash
# Main processing pipeline
python rmsai_lstm_autoencoder_proc.py --input data/ --output processed/
```

### Individual Component Testing
```bash
# Test database data loading
python test_dashboard_data.py

# Test anomaly alert logic
python test_anomaly_alert.py
```

---

## 3. Vector Database & Similarity Search

### ChromaDB Vector Database Status
```bash
python3 -c "
import chromadb
import numpy as np

# Connect to vector database
client = chromadb.PersistentClient(path='vector_db')
collection = client.get_collection('rmsai_ecg_embeddings')

# Get statistics
results = collection.get(limit=10)
total_vectors = len(results['ids']) if results['ids'] else 0

print('=== VECTOR DATABASE STATUS ===')
print(f'Total embeddings: {collection.count()}')
print(f'Collection name: {collection.name}')
print(f'Sample vector IDs: {results[\"ids\"][:5] if results[\"ids\"] else \"None\"}')

# Check embedding dimensions
if results['embeddings']:
    embedding_dim = len(results['embeddings'][0])
    print(f'Embedding dimensions: {embedding_dim}')

    # Calculate some statistics
    embeddings = np.array(results['embeddings'][:10])
    print(f'Average embedding norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}')
"
```

### Similarity Search Demo
```bash
# Get a sample chunk ID
CHUNK_ID=$(sqlite3 rmsai_metadata.db "SELECT chunk_id FROM chunks LIMIT 1;")

# Test similarity search via API
curl -X POST http://localhost:8000/api/v1/search/similar \
  -H "Content-Type: application/json" \
  -d "{\"chunk_id\": \"$CHUNK_ID\", \"n_results\": 5, \"include_metadata\": true}" \
  | python3 -m json.tool
```

---

## 4. API Server Demonstrations

### Start API Server
```bash
# Start the API server in background
python api_server.py &
sleep 3  # Wait for startup

# Check server status
curl -s http://localhost:8000/api/v1/stats | python3 -m json.tool
```

### API Endpoint Testing
```bash
# Get system statistics
echo "=== SYSTEM STATISTICS ==="
curl -s http://localhost:8000/api/v1/stats | python3 -m json.tool

# Get recent anomalies
echo -e "\n=== RECENT ANOMALIES ==="
curl -s "http://localhost:8000/api/v1/anomalies?limit=5" | python3 -m json.tool

# Get anomalies by condition
echo -e "\n=== BRADYCARDIA CASES ==="
curl -s "http://localhost:8000/api/v1/anomalies?condition=Bradycardia&limit=3" | python3 -m json.tool

# Get anomalies by lead
echo -e "\n=== ECG1 LEAD ANOMALIES ==="
curl -s "http://localhost:8000/api/v1/anomalies?lead=ECG1&limit=3" | python3 -m json.tool
```

---

## 5. Advanced Analytics Demonstrations

### Clustering Analysis
```bash
python3 -c "
from advanced_analytics import EmbeddingAnalytics
import json

analytics = EmbeddingAnalytics('vector_db', 'rmsai_metadata.db')

print('=== CLUSTERING ANALYSIS ===')
clusters = analytics.discover_embedding_clusters()

if 'kmeans' in clusters:
    print(f'K-means found {clusters[\"kmeans\"][\"n_clusters\"]} clusters')
    print(f'Silhouette score: {clusters[\"kmeans\"][\"silhouette_score\"]}')

if 'dbscan' in clusters:
    print(f'DBSCAN found {clusters[\"dbscan\"][\"n_clusters\"]} clusters')
    print(f'Noise points: {clusters[\"dbscan\"][\"noise_points\"]}')
"
```

### Anomaly Detection Analysis
```bash
python3 -c "
from advanced_analytics import EmbeddingAnalytics

analytics = EmbeddingAnalytics('vector_db', 'rmsai_metadata.db')

print('=== ANOMALY DETECTION METHODS ===')
anomalies = analytics.detect_anomalous_patterns(contamination=0.1)

for method in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
    if method in anomalies and 'n_anomalies' in anomalies[method]:
        print(f'{method.replace(\"_\", \" \").title()}: {anomalies[method][\"n_anomalies\"]} anomalies')

if 'consensus' in anomalies:
    print(f'Consensus anomalies: {anomalies[\"consensus\"][\"n_anomalies\"]}')
    print(f'Methods used: {anomalies[\"consensus\"][\"methods_used\"]}')
"
```

### Temporal Pattern Analysis
```bash
python3 -c "
from advanced_analytics import EmbeddingAnalytics

analytics = EmbeddingAnalytics('vector_db', 'rmsai_metadata.db')

print('=== TEMPORAL PATTERNS ===')
temporal = analytics.temporal_pattern_analysis()

if 'volume_trends' in temporal:
    print(f'Average daily volume: {temporal[\"volume_trends\"][\"avg_daily_volume\"]}')
    print(f'Total days analyzed: {temporal[\"volume_trends\"][\"total_days\"]}')
    print(f'Peak processing day: {temporal[\"volume_trends\"][\"peak_day\"]}')

if 'condition_temporal' in temporal and 'peak_hours' in temporal['condition_temporal']:
    print('\nPeak hours by condition:')
    for condition, hour in temporal['condition_temporal']['peak_hours'].items():
        print(f'  {condition}: {hour}:00')
"
```

---

## 6. Detailed Database Queries

### Recent Processing Activity
```bash
sqlite3 rmsai_metadata.db "
SELECT
    'Recent Processing Activity (Last 10):' as header,
    '' as chunk_id, '' as condition, '' as lead, '' as error_score, '' as timestamp
UNION ALL
SELECT
    '',
    chunk_id,
    COALESCE(JSON_EXTRACT(metadata, '$.condition'), 'Unknown') as condition,
    lead_name,
    ROUND(error_score, 4) as error_score,
    processing_timestamp
FROM chunks
ORDER BY processing_timestamp DESC
LIMIT 10;
"
```

### Error Score Analysis
```bash
sqlite3 rmsai_metadata.db "
SELECT
    'Error Score Distribution:' as range, '' as count
UNION ALL
SELECT
    CASE
        WHEN error_score < 0.3 THEN 'Low (< 0.3)'
        WHEN error_score < 0.6 THEN 'Medium (0.3-0.6)'
        WHEN error_score < 0.9 THEN 'High (0.6-0.9)'
        ELSE 'Very High (≥ 0.9)'
    END as range,
    COUNT(*) as count
FROM chunks
GROUP BY
    CASE
        WHEN error_score < 0.3 THEN 1
        WHEN error_score < 0.6 THEN 2
        WHEN error_score < 0.9 THEN 3
        ELSE 4
    END
ORDER BY range;
"
```

### Cross-Lead Correlation Analysis
```bash
sqlite3 rmsai_metadata.db "
SELECT
    'Cross-Lead Analysis by Event:' as header, '' as event, '' as leads, '' as conditions
UNION ALL
SELECT
    '',
    event_id,
    GROUP_CONCAT(DISTINCT lead_name) as leads,
    GROUP_CONCAT(DISTINCT JSON_EXTRACT(metadata, '$.condition')) as conditions
FROM chunks
GROUP BY event_id
HAVING COUNT(DISTINCT lead_name) > 1
LIMIT 10;
"
```

---

## 7. One-Line Demo Commands

### Quick System Health Check
```bash
echo "System Health:" && sqlite3 rmsai_metadata.db "SELECT COUNT(*) || ' chunks processed, ' || ROUND(AVG(error_score), 3) || ' avg error score, ' || COUNT(DISTINCT JSON_EXTRACT(metadata, '$.condition')) || ' conditions detected' FROM chunks;"
```

### Processing Rate Calculation
```bash
sqlite3 rmsai_metadata.db "SELECT ROUND(COUNT(*) / ((julianday(MAX(processing_timestamp)) - julianday(MIN(processing_timestamp))) * 24), 0) || ' chunks/hour processing rate' FROM chunks;"
```

### Anomaly Rate Summary
```bash
sqlite3 rmsai_metadata.db "SELECT ROUND(100.0 * COUNT(CASE WHEN anomaly_status='anomaly' THEN 1 END) / COUNT(*), 1) || '% anomaly detection rate' FROM chunks;"
```

---

## 8. Complete Demo Script

### Create and Run Complete Demo
```bash
# Create a comprehensive demo script
cat > demo_rmsai.sh << 'EOF'
#!/bin/bash
echo "=== RMSAI SYSTEM DEMONSTRATION ==="
echo

echo "1. System Overview:"
sqlite3 rmsai_metadata.db "SELECT 'Processed: ' || COUNT(*) || ' chunks, ' || COUNT(DISTINCT event_id) || ' events, ' || COUNT(DISTINCT lead_name) || ' leads' FROM chunks;"
echo

echo "2. Condition Distribution:"
sqlite3 rmsai_metadata.db "SELECT JSON_EXTRACT(metadata, '$.condition') as condition, COUNT(*) as count FROM chunks GROUP BY condition ORDER BY count DESC;"
echo

echo "3. Processing Statistics:"
sqlite3 rmsai_metadata.db "SELECT 'Avg Error Score: ' || ROUND(AVG(error_score), 4), 'Anomaly Rate: ' || ROUND(100.0 * SUM(CASE WHEN anomaly_status='anomaly' THEN 1 ELSE 0 END) / COUNT(*), 1) || '%' FROM chunks;"
echo

echo "4. Recent Activity:"
sqlite3 rmsai_metadata.db "SELECT chunk_id, lead_name, ROUND(error_score, 3), processing_timestamp FROM chunks ORDER BY processing_timestamp DESC LIMIT 5;"
echo

echo "5. API Status:"
curl -s http://localhost:8000/api/v1/stats | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'API: {data[\"total_chunks\"]} chunks, {data[\"anomaly_rate\"]}% anomaly rate')"
EOF

chmod +x demo_rmsai.sh
./demo_rmsai.sh
```

This comprehensive guide provides all the command-line tools you need to demonstrate every aspect of the RMSAI system without using dashboards!