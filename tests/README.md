# RMSAI Test Suite

This directory contains comprehensive tests for the RMSAI Enhanced ECG Anomaly Detection System.

## Test Files

### `test_improvements.py`
**Comprehensive test suite for all enhancement modules:**

- ✅ **API Server Tests**: REST endpoints, WebSocket connectivity, health checks
- ✅ **Advanced Analytics Tests**: ML clustering, anomaly detection, visualization
- ✅ **Adaptive Thresholds Tests**: Statistical optimization, performance evaluation
- ✅ **Threshold Classification Tests**: Heart rate-based anomaly classification, threshold validation
- ✅ **Monitoring Dashboard Tests**: Data loading, visualization components
- ✅ **Pacer Functionality Tests**: HDF5 pacer data validation, analysis functions
- ✅ **LSTM Processor Tests**: Chunking logic, configuration validation, database integration
- ✅ **Integration Tests**: Component interaction and data flow

#### Usage:
```bash
# Test all improvements
python tests/test_improvements.py

# Test specific component
python tests/test_improvements.py api
python tests/test_improvements.py analytics
python tests/test_improvements.py thresholds
python tests/test_improvements.py classification
python tests/test_improvements.py dashboard
python tests/test_improvements.py pacer
python tests/test_improvements.py processor

# Verbose output
python tests/test_improvements.py --verbose
```

### `test_threshold_classification.py`
**Comprehensive threshold and heart rate-based classification tests:**

- ✅ **Threshold Values**: Validates optimized threshold configuration (0.8-1.0 range)
- ✅ **Heart Rate Ranges**: Tests clinical HR ranges (Brady ≤60, Tachy ≥100 BPM)
- ✅ **Classification Logic**: Heart rate-based Tachy/Brady disambiguation
- ✅ **Threshold Hierarchy**: Validates clinical logic (Normal < Tachy/Brady < A-Fib < V-Tac)
- ✅ **Clinical Accuracy**: Simulates real scenarios with observed data patterns

#### Usage:
```bash
# Run threshold classification tests (standalone)
python tests/test_threshold_classification.py

# Run via main test suite
python tests/test_improvements.py classification
```

### `test_processor.py`
**Core LSTM processor functionality tests:**

- Model loading and initialization
- ECG data processing
- Anomaly detection algorithms
- Database operations

#### Usage:
```bash
# Run processor tests
python tests/test_processor.py
```

## Running Tests

### All Tests
```bash
# From project root
python -m pytest tests/

# Or run individual files
python tests/test_improvements.py
python tests/test_processor.py
```

### Test Coverage
The test suite covers:
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Cross-component communication
- **Performance Tests**: Response times and throughput
- **Error Handling**: Graceful degradation scenarios
- **Data Validation**: Input/output format verification

### Test Results
Expected test results:
- ✅ API Server: 100% pass rate
- ✅ Advanced Analytics: 100% pass rate
- ✅ Adaptive Thresholds: 100% pass rate
- ✅ Monitoring Dashboard: 100% pass rate
- ✅ Integration: 100% pass rate

## Dependencies

Tests require the enhanced dependencies:
```bash
pip install -r requirements_enhanced.txt
```

## Continuous Integration

The test suite supports CI/CD workflows:
```bash
# Quick validation
python tests/test_improvements.py --quick

# Full test suite
python tests/test_improvements.py --comprehensive

# Performance benchmarks
python tests/test_improvements.py --benchmark
```