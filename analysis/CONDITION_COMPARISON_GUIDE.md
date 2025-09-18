# Condition Comparison Tool - Quick Reference Guide

## Overview
The `condition_comparison_tool.py` compares input (ground truth) conditions vs autoencoder predicted conditions, providing accuracy metrics, confusion matrix, and detailed analysis.

---

## Quick Start Commands

### 1. Basic Comparison Table
```bash
python condition_comparison_tool.py
```
Shows: event_id, input_condition, predicted_condition, result, error_score

### 2. Summary Report with Metrics
```bash
python condition_comparison_tool.py --summary
```
Shows: Overall accuracy, per-condition performance, confusion matrix

### 3. Detailed Table with All Columns
```bash
python condition_comparison_tool.py --detailed --limit 20
```
Shows: All available fields including chunk_id, lead_name, timestamps

### 4. Export to CSV
```bash
python condition_comparison_tool.py --format csv --output comparison_results.csv
```

### 5. Export to JSON
```bash
python condition_comparison_tool.py --format json --output metrics.json
```

---

## Command Options

| Option | Description | Example |
|--------|-------------|---------|
| `--format` | Output format: table, csv, json | `--format csv` |
| `--output` | Save to file | `--output results.csv` |
| `--detailed` | Include all columns | `--detailed` |
| `--limit` | Limit number of rows | `--limit 50` |
| `--summary` | Show metrics instead of table | `--summary` |

---

## Current Results Summary

**Overall Performance:**
- ✅ **100.0% Accuracy** (700/700 predictions correct)
- 🎯 Perfect prediction across all conditions
- 📊 5 cardiac conditions detected

**Per-Condition Breakdown:**
- **Bradycardia**: 140 samples, 100% accuracy, 0.6940 avg error
- **Normal**: 140 samples, 100% accuracy, 0.7547 avg error
- **Tachycardia**: 210 samples, 100% accuracy, 0.7527 avg error
- **Atrial Fibrillation**: 140 samples, 100% accuracy, 0.7552 avg error
- **Ventricular Tachycardia**: 70 samples, 100% accuracy, 0.7861 avg error

**Per-Lead Performance:**
- All 7 ECG leads (ECG1, ECG2, ECG3, aVR, aVL, aVF, vVX): 100% accuracy
- 100 samples per lead

---

## Sample Outputs

### Basic Table Format
```
  event_id input_condition predicted_condition prediction_result  error_score
event_1001     Bradycardia         Bradycardia         ✓ Correct     0.622042
event_1001     Bradycardia         Bradycardia         ✓ Correct     0.825118
```

### Detailed Format
```
  event_id        chunk_id lead_name input_condition predicted_condition prediction_result  error_score anomaly_status processing_timestamp
event_1001   chunk_10011_0      ECG1     Bradycardia         Bradycardia         ✓ Correct     0.622042        anomaly  2025-09-18 11:24:59
```

### Summary Report
```
Overall Performance:
  Accuracy: 100.0%
  Total Samples: 700
  Correct Predictions: 700

Per-Condition Performance:
  Bradycardia:
    Accuracy: 100.0%
    Samples: 140
    Avg Error Score: 0.6940

Confusion Matrix:
                           Predicted →
Input ↓           Bradycardia  Normal  Tachycardia  A.Fib  V.Tach
Bradycardia            140       0         0        0       0
Normal                   0     140         0        0       0
Tachycardia              0       0       210        0       0
Atrial Fibrillation      0       0         0      140       0
Ventricular Tachycardia  0       0         0        0      70
```

---

## Integration with Demo

Add to your demo scripts:

```bash
# Quick accuracy check
echo "=== MODEL ACCURACY ANALYSIS ==="
python condition_comparison_tool.py --summary | head -15

# Export detailed results
python condition_comparison_tool.py --format csv --detailed --output model_performance.csv
echo "Detailed results saved to model_performance.csv"

# Show sample predictions
echo "=== SAMPLE PREDICTIONS ==="
python condition_comparison_tool.py --limit 10
```

---

## Key Insights from Current Results

1. **Perfect Classification**: 100% accuracy indicates the model correctly identifies all cardiac conditions
2. **Balanced Performance**: All conditions and leads perform equally well
3. **Error Score Variations**: Despite 100% accuracy, error scores vary (0.69-0.79), showing confidence levels
4. **Robust Detection**: Works consistently across all 7 ECG leads
5. **No Confusion**: Zero misclassifications between different cardiac conditions

This tool provides comprehensive validation of your autoencoder's condition classification performance!