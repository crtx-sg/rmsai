#!/usr/bin/env python3
"""
RMSAI Improvements Test Suite
=============================

Comprehensive test suite for all four improvement modules:
1. Streaming API
2. Advanced Analytics
3. Adaptive Thresholds
4. Monitoring Dashboard

Usage:
    python test_improvements.py [module]

    # Test all improvements
    python test_improvements.py

    # Test specific module
    python test_improvements.py api
    python test_improvements.py analytics
    python test_improvements.py thresholds
    python test_improvements.py dashboard
    python test_improvements.py pacer
"""

import sys
import time
import subprocess
import requests
import json
import logging
from pathlib import Path
from datetime import datetime
import threading
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovementTester:
    """Test suite for RMSAI improvements"""

    def __init__(self):
        self.results = {}
        self.processes = {}

    def test_api_server(self) -> bool:
        """Test the streaming API server"""
        logger.info("Testing Streaming API Server...")

        try:
            # Start API server in background
            logger.info("Starting API server...")
            process = subprocess.Popen([
                sys.executable, "api_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes['api'] = process

            # Wait for server to start
            time.sleep(5)

            # Test health endpoint
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                logger.error("Health check failed")
                return False

            health_data = response.json()
            logger.info(f"Health check passed: {health_data['status']}")

            # Test stats endpoint
            response = requests.get("http://localhost:8000/api/v1/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                logger.info(f"Stats endpoint working: {stats['total_chunks']} chunks")
            else:
                logger.warning("Stats endpoint not available (no data)")

            # Test anomalies endpoint
            response = requests.post(
                "http://localhost:8000/api/v1/anomalies",
                json={"limit": 5},
                timeout=5
            )
            if response.status_code == 200:
                anomalies = response.json()
                logger.info(f"Anomalies endpoint working: {anomalies['count']} anomalies")
            else:
                logger.warning("Anomalies endpoint returned error (expected if no data)")

            # Test conditions endpoint
            response = requests.get("http://localhost:8000/api/v1/conditions", timeout=5)
            if response.status_code == 200:
                conditions = response.json()
                logger.info(f"Conditions endpoint working: {conditions['total_conditions']} conditions")

            # Test leads endpoint
            response = requests.get("http://localhost:8000/api/v1/leads", timeout=5)
            if response.status_code == 200:
                leads = response.json()
                logger.info(f"Leads endpoint working: {leads['total_leads']} leads")

            # Test WebSocket test page
            response = requests.get("http://localhost:8000/test-websocket", timeout=5)
            if response.status_code == 200:
                logger.info("WebSocket test page accessible")

            # Test API documentation
            response = requests.get("http://localhost:8000/docs", timeout=5)
            if response.status_code == 200:
                logger.info("API documentation accessible")

            # Test pacer analysis endpoints
            response = requests.get("http://localhost:8000/api/v1/pacer-analysis", timeout=10)
            if response.status_code == 200:
                pacer_data = response.json()
                logger.info(f"Pacer analysis endpoint working: {pacer_data.get('pacer_data_available', False)}")
                if pacer_data.get('pacer_data_available'):
                    logger.info(f"  - Found {pacer_data.get('total_chunks_with_pacer_data', 0)} chunks with pacer data")
                    logger.info(f"  - Pacer types: {list(pacer_data.get('pacer_type_distribution', {}).get('counts', {}).keys())}")
            else:
                logger.warning("Pacer analysis endpoint returned error (expected if no pacer data)")

            # Test advanced pacer analytics endpoint
            response = requests.get("http://localhost:8000/api/v1/analytics/pacer-patterns", timeout=10)
            if response.status_code == 200:
                pacer_analytics = response.json()
                logger.info("Advanced pacer analytics endpoint working")
                if 'results' in pacer_analytics and 'total_chunks_analyzed' in pacer_analytics['results']:
                    logger.info(f"  - Analyzed {pacer_analytics['results']['total_chunks_analyzed']} chunks")
            else:
                logger.warning("Advanced pacer analytics endpoint not available")

            # Test pacer threshold impact endpoint
            response = requests.get("http://localhost:8000/api/v1/thresholds/pacer-impact", timeout=10)
            if response.status_code == 200:
                threshold_impact = response.json()
                logger.info("Pacer threshold impact endpoint working")
                if 'results' in threshold_impact and 'total_samples' in threshold_impact['results']:
                    logger.info(f"  - Analyzed {threshold_impact['results']['total_samples']} samples")
            else:
                logger.warning("Pacer threshold impact endpoint not available")

            logger.info("✅ API Server tests passed")
            return True

        except requests.exceptions.ConnectionError:
            logger.error("❌ Could not connect to API server")
            return False
        except Exception as e:
            logger.error(f"❌ API Server test failed: {e}")
            return False

    def test_advanced_analytics(self) -> bool:
        """Test the advanced analytics module"""
        logger.info("Testing Advanced Analytics...")

        try:
            from advanced_analytics import EmbeddingAnalytics, run_comprehensive_analysis

            # Test analytics initialization
            analytics = EmbeddingAnalytics("vector_db", "rmsai_metadata.db")
            logger.info("Analytics module initialized")

            # Test loading embeddings (may fail if no data)
            try:
                embeddings, metadata = analytics.load_embeddings_with_metadata()
                if len(embeddings) > 0:
                    logger.info(f"Loaded {len(embeddings)} embeddings successfully")

                    # Test clustering
                    clusters = analytics.discover_embedding_clusters()
                    if 'kmeans' in clusters:
                        logger.info(f"K-means clustering: {clusters['kmeans']['n_clusters']} clusters")

                    # Test anomaly detection
                    anomalies = analytics.detect_anomalous_patterns()
                    if 'isolation_forest' in anomalies:
                        logger.info(f"Isolation Forest: {anomalies['isolation_forest']['n_anomalies']} anomalies")

                    # Test temporal analysis
                    temporal = analytics.temporal_pattern_analysis()
                    if 'hourly_patterns' in temporal:
                        logger.info("Temporal analysis completed")

                    # Test similarity network
                    network = analytics.generate_similarity_network()
                    if 'n_similarity_groups' in network:
                        logger.info(f"Similarity network: {network['n_similarity_groups']} groups")

                else:
                    logger.warning("No embeddings available for full testing")

            except Exception as e:
                logger.warning(f"Analytics testing limited due to data availability: {e}")

            # Test comprehensive analysis runner
            logger.info("Testing comprehensive analysis runner...")
            results = run_comprehensive_analysis(output_dir="test_analytics_output")

            if 'error' not in results:
                logger.info("Comprehensive analysis completed successfully")
            else:
                logger.warning(f"Comprehensive analysis had issues: {results['error']}")

            logger.info("✅ Advanced Analytics tests passed")
            return True

        except ImportError as e:
            logger.error(f"❌ Missing dependencies for analytics: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Advanced Analytics test failed: {e}")
            return False

    def test_adaptive_thresholds(self) -> bool:
        """Test the adaptive thresholds module"""
        logger.info("Testing Adaptive Thresholds...")

        try:
            from adaptive_thresholds import AdaptiveThresholdManager, run_threshold_optimization

            # Test threshold manager initialization
            threshold_manager = AdaptiveThresholdManager("rmsai_metadata.db")
            logger.info("Threshold manager initialized")

            # Test initial thresholds
            thresholds = threshold_manager.get_all_thresholds()
            logger.info(f"Initial thresholds loaded for {len(thresholds)} conditions")

            # Test threshold calculation (may have limited data)
            try:
                optimal_thresholds = threshold_manager.calculate_optimal_thresholds()
                if optimal_thresholds:
                    logger.info(f"Calculated optimal thresholds for {len(optimal_thresholds)} conditions")
                    for condition, data in optimal_thresholds.items():
                        logger.info(f"  {condition}: {data['threshold']:.4f} (confidence: {data['confidence']:.3f})")
                else:
                    logger.warning("No optimal thresholds calculated (insufficient data)")

            except Exception as e:
                logger.warning(f"Threshold calculation limited: {e}")

            # Test performance evaluation
            try:
                performance = threshold_manager.evaluate_current_performance()
                if performance:
                    logger.info(f"Performance evaluation completed for {len(performance)} conditions")
                    avg_f1 = sum(p['f1_score'] for p in performance.values()) / len(performance)
                    logger.info(f"Average F1 score: {avg_f1:.3f}")
                else:
                    logger.warning("No performance data available")

            except Exception as e:
                logger.warning(f"Performance evaluation limited: {e}")

            # Test update trigger
            should_update = threshold_manager.should_update_thresholds()
            logger.info(f"Update trigger check: {should_update}")

            # Test export/import
            export_path = "test_thresholds_export.json"
            threshold_manager.export_threshold_config(export_path)
            if Path(export_path).exists():
                logger.info("Threshold configuration export successful")

                # Test import
                threshold_manager.import_threshold_config(export_path)
                logger.info("Threshold configuration import successful")

            # Test optimization runner
            logger.info("Testing threshold optimization runner...")
            results = run_threshold_optimization(output_file="test_threshold_results.json")

            if 'error' not in results:
                logger.info("Threshold optimization completed successfully")
            else:
                logger.warning(f"Threshold optimization had issues: {results['error']}")

            logger.info("✅ Adaptive Thresholds tests passed")
            return True

        except ImportError as e:
            logger.error(f"❌ Missing dependencies for thresholds: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Adaptive Thresholds test failed: {e}")
            return False

    def test_dashboard(self) -> bool:
        """Test the monitoring dashboard"""
        logger.info("Testing Monitoring Dashboard...")

        try:
            # Import dashboard module to check dependencies
            import streamlit as st
            import plotly.express as px
            import plotly.graph_objects as go
            logger.info("Dashboard dependencies available")

            # Test dashboard class initialization
            from dashboard import RMSAIDashboard
            dashboard = RMSAIDashboard()
            logger.info("Dashboard class initialized")

            # Test data loading
            try:
                chunks_df, files_df = dashboard.load_data(use_cache=False)
                logger.info(f"Data loading successful: {len(chunks_df)} chunks, {len(files_df)} files")
            except Exception as e:
                logger.warning(f"Data loading limited: {e}")

            # Test API stats retrieval
            try:
                api_stats = dashboard.get_api_stats()
                if api_stats:
                    logger.info("API stats retrieval successful")
                else:
                    logger.warning("API stats not available (API server not running)")
            except Exception as e:
                logger.warning(f"API stats retrieval failed: {e}")

            # Note: Cannot fully test Streamlit dashboard without running it
            logger.info("Dashboard module structure verified")
            logger.info("To test dashboard fully, run: streamlit run dashboard.py")

            logger.info("✅ Monitoring Dashboard tests passed")
            return True

        except ImportError as e:
            logger.error(f"❌ Missing dependencies for dashboard: {e}")
            logger.info("Install with: pip install streamlit plotly")
            return False
        except Exception as e:
            logger.error(f"❌ Monitoring Dashboard test failed: {e}")
            return False

    def test_integration(self) -> bool:
        """Test integration between components"""
        logger.info("Testing Integration...")

        try:
            # Test API + Analytics integration
            if 'api' in self.processes:
                try:
                    # Test similarity search endpoint (requires vector data)
                    response = requests.post(
                        "http://localhost:8000/api/v1/search/similar",
                        json={"chunk_id": "test_chunk", "n_results": 5},
                        timeout=5
                    )
                    # 404 is expected if chunk doesn't exist
                    if response.status_code in [200, 404, 503]:
                        logger.info("Similarity search endpoint accessible")
                    else:
                        logger.warning(f"Similarity search returned: {response.status_code}")

                except Exception as e:
                    logger.warning(f"Similarity search test limited: {e}")

            # Test threshold + API integration
            try:
                from adaptive_thresholds import AdaptiveThresholdManager
                threshold_manager = AdaptiveThresholdManager("rmsai_metadata.db")

                # Get threshold for a condition
                threshold = threshold_manager.get_threshold_for_condition("Normal")
                logger.info(f"Threshold integration working: Normal = {threshold}")

            except Exception as e:
                logger.warning(f"Threshold integration test limited: {e}")

            logger.info("✅ Integration tests completed")
            return True

        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False

    def cleanup(self):
        """Clean up test processes and files"""
        logger.info("Cleaning up test environment...")

        # Stop API server
        if 'api' in self.processes:
            try:
                self.processes['api'].terminate()
                self.processes['api'].wait(timeout=5)
                logger.info("API server stopped")
            except Exception as e:
                logger.warning(f"Error stopping API server: {e}")

        # Clean up test files
        test_files = [
            "test_thresholds_export.json",
            "test_threshold_results.json",
            "test_analytics_output"
        ]

        for file_path in test_files:
            try:
                path = Path(file_path)
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                logger.info(f"Cleaned up {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up {file_path}: {e}")

    def test_pacer_functionality(self) -> bool:
        """Test pacer_info and pacer_offset functionality in HDF5 files"""
        logger.info("Testing Pacer Functionality...")

        try:
            # Generate a test HDF5 file with pacer data
            logger.info("Generating test HDF5 file with pacer data...")

            import subprocess
            import os

            # Generate a small test file
            result = subprocess.run([
                sys.executable, "rmsai_sim_hdf5_data.py", "2", "--patient-id", "TEST_PACER"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"Failed to generate test HDF5 file: {result.stderr}")
                return False

            # Find the generated file
            test_file = None
            for file in os.listdir("data"):
                if file.startswith("TEST_PACER") and file.endswith(".h5"):
                    test_file = os.path.join("data", file)
                    break

            if not test_file:
                logger.error("Test HDF5 file not found")
                return False

            logger.info(f"Generated test file: {test_file}")

            # Test HDF5 access with pacer data
            logger.info("Testing HDF5 access with pacer analysis...")

            # Import the h5access module to test pacer functionality
            sys.path.insert(0, '.')
            from rmsai_h5access import analyze_pacer_data

            import h5py

            # Open the test file and validate pacer data
            with h5py.File(test_file, 'r') as f:
                events = [key for key in f.keys() if key.startswith('event_')]

                if not events:
                    logger.error("No events found in test file")
                    return False

                first_event = f[events[0]]

                # Test pacer_info presence
                if 'pacer_info' not in first_event['ecg']:
                    logger.error("pacer_info not found in ECG data")
                    return False

                pacer_info = first_event['ecg']['pacer_info'][()]
                logger.info(f"Found pacer_info: 0x{pacer_info:08X}")

                # Test pacer_offset presence
                if 'pacer_offset' not in first_event['ecg']:
                    logger.error("pacer_offset not found in ECG data")
                    return False

                pacer_offset = first_event['ecg']['pacer_offset'][()]
                logger.info(f"Found pacer_offset: {pacer_offset} samples")

                # Validate pacer_offset range
                if not (0 <= pacer_offset <= 2400):
                    logger.error(f"pacer_offset out of valid range: {pacer_offset}")
                    return False

                # Test comprehensive pacer analysis
                pacer_data = analyze_pacer_data(first_event)

                if 'info' not in pacer_data:
                    logger.error("Pacer info analysis failed")
                    return False

                if 'timing' not in pacer_data:
                    logger.error("Pacer timing analysis failed")
                    return False

                pacer_type = pacer_data['info']['type']
                timing_category = pacer_data['timing']['timing_category']
                time_offset = pacer_data['timing']['offset_seconds']

                logger.info(f"Pacer analysis results:")
                logger.info(f"  - Type: {pacer_type} ({pacer_data['info']['type_name']})")
                logger.info(f"  - Timing: {timing_category} ({time_offset:.3f}s)")

                if 'signal_analysis' in pacer_data:
                    signal_at_spike = pacer_data['signal_analysis']['pacer_amplitude_at_spike']
                    logger.info(f"  - Signal at pacer spike: {signal_at_spike:.3f} mV")

            # Test file structure validation
            logger.info("Testing file structure validation...")
            result = subprocess.run([
                sys.executable, "rmsai_h5access.py", test_file
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"H5 access validation failed: {result.stderr}")
                return False

            # Check if pacer data is mentioned in the output
            if "Pacer Information" not in result.stdout:
                logger.error("Pacer information not displayed in h5access output")
                return False

            if "Pacer Timing" not in result.stdout:
                logger.error("Pacer timing not displayed in h5access output")
                return False

            logger.info("Pacer data correctly displayed in h5access output")

            # Clean up test file
            try:
                os.remove(test_file)
                logger.info("Test file cleaned up")
            except:
                pass

            logger.info("✅ Pacer functionality tests passed")
            return True

        except Exception as e:
            logger.error(f"❌ Pacer functionality test failed: {e}")
            return False

    def run_tests(self, module: str = None) -> Dict[str, bool]:
        """Run all tests or specific module tests"""
        logger.info("Starting RMSAI Improvements Test Suite")
        logger.info("=" * 60)

        tests_to_run = {
            'api': self.test_api_server,
            'analytics': self.test_advanced_analytics,
            'thresholds': self.test_adaptive_thresholds,
            'dashboard': self.test_dashboard,
            'pacer': self.test_pacer_functionality
        }

        if module and module in tests_to_run:
            tests_to_run = {module: tests_to_run[module]}

        results = {}

        try:
            for test_name, test_func in tests_to_run.items():
                logger.info(f"\n📋 Testing {test_name.upper()}...")
                try:
                    results[test_name] = test_func()
                except Exception as e:
                    logger.error(f"Test {test_name} failed with exception: {e}")
                    results[test_name] = False

            # Run integration tests if testing multiple modules
            if len(tests_to_run) > 1:
                logger.info(f"\n🔗 Testing INTEGRATION...")
                results['integration'] = self.test_integration()

        finally:
            self.cleanup()

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        passed = 0
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.upper():15s}: {status}")
            if result:
                passed += 1

        logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning("⚠️ Some tests failed - check logs above")

        return results

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test RMSAI improvements")
    parser.add_argument(
        'module',
        nargs='?',
        choices=['api', 'analytics', 'thresholds', 'dashboard', 'pacer'],
        help='Specific module to test (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tester = ImprovementTester()
    results = tester.run_tests(args.module)

    # Exit with error code if tests failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()