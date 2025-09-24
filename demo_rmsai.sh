#!/bin/bash
echo "=== RMSAI SYSTEM DEMONSTRATION ==="
echo

echo "1. System Overview:"
sqlite3 rmsai_metadata.db "SELECT 'Processed: ' || COUNT(*) || ' chunks, ' || COUNT(DISTINCT event_id) || ' events, ' || COUNT(DISTINCT lead_name) || ' leads' FROM chunks;"
echo

echo "2. Condition Distribution:"
sqlite3 rmsai_metadata.db "SELECT '  ' || JSON_EXTRACT(metadata, '$.condition') || ': ' || COUNT(*) || ' chunks' FROM chunks GROUP BY JSON_EXTRACT(metadata, '$.condition') ORDER BY COUNT(*) DESC;"
echo

echo "3. Processing Statistics:"
sqlite3 rmsai_metadata.db "SELECT '  Avg Error Score: ' || ROUND(AVG(error_score), 4) FROM chunks;"
sqlite3 rmsai_metadata.db "SELECT '  Anomaly Rate: ' || ROUND(100.0 * SUM(CASE WHEN anomaly_status='anomaly' THEN 1 ELSE 0 END) / COUNT(*), 1) || '%' FROM chunks;"
echo

echo "4. Recent Activity:"
sqlite3 rmsai_metadata.db "SELECT '  ' || chunk_id || ' (' || lead_name || ') - Error: ' || ROUND(error_score, 3) || ' - ' || processing_timestamp FROM chunks ORDER BY processing_timestamp DESC LIMIT 5;"
echo

echo "5. API Status:"
if curl -s --connect-timeout 2 http://localhost:8000/api/v1/stats > /tmp/api_response.json 2>/dev/null; then
    if python3 -c "
import sys, json
try:
    with open('/tmp/api_response.json', 'r') as f:
        data = json.load(f)
    print(f'API: {data[\"total_chunks\"]} chunks, {data[\"anomaly_rate\"]}% anomaly rate')
except (json.JSONDecodeError, KeyError) as e:
    print('API: Error parsing response')
    exit(1)
"; then
        echo "API service is running"
    else
        echo "API: Response parsing failed"
    fi
else
    echo "API: Service not available (connection failed)"
fi
rm -f /tmp/api_response.json
