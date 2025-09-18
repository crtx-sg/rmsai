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
