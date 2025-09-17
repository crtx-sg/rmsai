#!/usr/bin/env python3
"""
RMSAI Adaptive Threshold Manager
================================

Dynamic threshold adjustment system that learns from data distribution and
clinical feedback to optimize anomaly detection accuracy.

Features:
- Statistical threshold optimization (GMM, percentile, ROC, PR curves)
- Condition-specific threshold adaptation
- Performance monitoring and feedback
- Smooth threshold updates with confidence weighting
- Real-time threshold adjustment based on new data
- Historical performance tracking

Usage:
    from adaptive_thresholds import AdaptiveThresholdManager

    threshold_manager = AdaptiveThresholdManager("rmsai_metadata.db")
    updated = threshold_manager.update_thresholds()
    threshold = threshold_manager.get_threshold_for_condition("Ventricular Tachycardia")
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from scipy import stats
from scipy.optimize import minimize_scalar
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveThresholdManager:
    """Dynamic threshold adjustment for optimal anomaly detection"""

    def __init__(self, sqlite_db_path: str, update_frequency: int = 100,
                 confidence_threshold: float = 0.6):
        self.sqlite_db_path = sqlite_db_path
        self.update_frequency = update_frequency  # Update after N new samples
        self.confidence_threshold = confidence_threshold  # Minimum confidence for updates
        self.thresholds = {}
        self.performance_history = []
        self.last_update_count = 0
        self.last_update_time = None

        # Statistical parameters
        self.smoothing_factors = {
            'high_confidence': 0.7,    # >0.8 confidence
            'medium_confidence': 0.5,  # 0.6-0.8 confidence
            'low_confidence': 0.3      # <0.6 confidence
        }
        self.min_samples_per_condition = 20
        self.max_threshold_change = 0.5  # Maximum relative change per update

        # Load initial thresholds
        self.load_initial_thresholds()

    def load_initial_thresholds(self):
        """Load initial condition-specific thresholds"""
        self.thresholds = {
            'Normal': {
                'threshold': 0.05,
                'confidence': 0.8,
                'last_updated': datetime.now().isoformat(),
                'update_count': 0,
                'performance_history': []
            },
            'Tachycardia': {
                'threshold': 0.08,
                'confidence': 0.8,
                'last_updated': datetime.now().isoformat(),
                'update_count': 0,
                'performance_history': []
            },
            'Bradycardia': {
                'threshold': 0.07,
                'confidence': 0.8,
                'last_updated': datetime.now().isoformat(),
                'update_count': 0,
                'performance_history': []
            },
            'Atrial Fibrillation (PTB-XL)': {
                'threshold': 0.12,
                'confidence': 0.8,
                'last_updated': datetime.now().isoformat(),
                'update_count': 0,
                'performance_history': []
            },
            'Ventricular Tachycardia (MIT-BIH)': {
                'threshold': 0.15,
                'confidence': 0.8,
                'last_updated': datetime.now().isoformat(),
                'update_count': 0,
                'performance_history': []
            },
            'Unknown': {
                'threshold': 0.10,
                'confidence': 0.7,
                'last_updated': datetime.now().isoformat(),
                'update_count': 0,
                'performance_history': []
            }
        }

        logger.info(f"Initialized thresholds for {len(self.thresholds)} conditions")

    def should_update_thresholds(self) -> bool:
        """Check if thresholds should be updated based on new data"""
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                current_count = cursor.fetchone()[0]

            # Check count-based trigger
            count_trigger = (current_count - self.last_update_count) >= self.update_frequency

            # Check time-based trigger (at least 1 hour since last update)
            time_trigger = False
            if self.last_update_time:
                time_since_update = datetime.now() - self.last_update_time
                time_trigger = time_since_update.total_seconds() > 3600  # 1 hour

            return count_trigger or time_trigger

        except Exception as e:
            logger.error(f"Error checking update trigger: {e}")
            return False

    def calculate_optimal_thresholds(self, condition: str = None,
                                   methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate optimal thresholds using multiple statistical methods"""
        if methods is None:
            methods = ['percentile', 'roc', 'pr_curve', 'gmm']

        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                if condition:
                    # Single condition query
                    query = """
                        SELECT error_score, anomaly_status, anomaly_type
                        FROM chunks
                        WHERE anomaly_type = ? OR (anomaly_type IS NULL AND ? = 'Unknown')
                    """
                    df = pd.read_sql_query(query, conn, params=(condition, condition))
                else:
                    # All conditions query
                    query = """
                        SELECT error_score, anomaly_status, anomaly_type
                        FROM chunks
                        WHERE processing_timestamp >= datetime('now', '-7 days')
                    """
                    df = pd.read_sql_query(query, conn)

        except Exception as e:
            logger.error(f"Error loading data for threshold calculation: {e}")
            return {}

        if len(df) == 0:
            logger.warning("No data available for threshold calculation")
            return {}

        optimal_thresholds = {}

        # Group by condition (handle NULL values)
        df['condition_clean'] = df['anomaly_type'].fillna('Unknown')

        for cond in df['condition_clean'].unique():
            condition_data = df[df['condition_clean'] == cond]

            if len(condition_data) < self.min_samples_per_condition:
                logger.info(f"Insufficient data for {cond}: {len(condition_data)} samples")
                continue

            logger.info(f"Calculating thresholds for {cond} with {len(condition_data)} samples")

            # Calculate thresholds using different methods
            method_results = {}

            for method in methods:
                try:
                    if method == 'percentile':
                        result = self._calculate_percentile_threshold(condition_data)
                    elif method == 'roc':
                        result = self._calculate_roc_threshold(condition_data)
                    elif method == 'pr_curve':
                        result = self._calculate_pr_threshold(condition_data)
                    elif method == 'gmm':
                        result = self._calculate_gmm_threshold(condition_data)
                    else:
                        continue

                    if result is not None:
                        method_results[method] = result

                except Exception as e:
                    logger.warning(f"Method {method} failed for {cond}: {e}")
                    continue

            if not method_results:
                logger.warning(f"No valid threshold calculations for {cond}")
                continue

            # Combine methods with weighted average
            optimal_threshold, confidence = self._combine_threshold_methods(
                method_results, condition_data
            )

            optimal_thresholds[cond] = {
                'threshold': float(optimal_threshold),
                'confidence': float(confidence),
                'methods_used': list(method_results.keys()),
                'method_values': method_results,
                'sample_count': len(condition_data),
                'calculated_at': datetime.now().isoformat()
            }

        return optimal_thresholds

    def _calculate_percentile_threshold(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate threshold using percentile method on normal data"""
        try:
            normal_scores = data[data['anomaly_status'] == 'normal']['error_score']

            if len(normal_scores) < 5:
                return None

            # Use 95th percentile of normal scores
            threshold = np.percentile(normal_scores, 95)
            return threshold

        except Exception as e:
            logger.debug(f"Percentile threshold calculation failed: {e}")
            return None

    def _calculate_roc_threshold(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate optimal threshold using ROC curve analysis"""
        try:
            y_true = (data['anomaly_status'] == 'anomaly').astype(int)
            y_scores = data['error_score']

            if len(np.unique(y_true)) < 2:
                return None

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)

            # Find threshold that maximizes Youden's J statistic (TPR - FPR)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]

            return optimal_threshold

        except Exception as e:
            logger.debug(f"ROC threshold calculation failed: {e}")
            return None

    def _calculate_pr_threshold(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate optimal threshold using Precision-Recall curve"""
        try:
            y_true = (data['anomaly_status'] == 'anomaly').astype(int)
            y_scores = data['error_score']

            if len(np.unique(y_true)) < 2:
                return None

            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)

            if len(f1_scores) > 0:
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
                return optimal_threshold

            return None

        except Exception as e:
            logger.debug(f"PR threshold calculation failed: {e}")
            return None

    def _calculate_gmm_threshold(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate threshold using Gaussian Mixture Model"""
        try:
            error_scores = data['error_score'].values.reshape(-1, 1)

            if len(error_scores) < 10:
                return None

            # Fit 2-component GMM (normal vs anomaly distributions)
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(error_scores)

            # Get component parameters
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            weights = gmm.weights_

            # Sort components by mean (low = normal, high = anomaly)
            sorted_indices = np.argsort(means)
            normal_mean = means[sorted_indices[0]]
            normal_std = np.sqrt(covariances[sorted_indices[0]])
            anomaly_mean = means[sorted_indices[1]]

            # Find intersection point of the two distributions
            # Use 3-sigma rule from normal distribution
            threshold = normal_mean + 3 * normal_std

            # Ensure threshold is between the means
            threshold = np.clip(threshold, normal_mean, anomaly_mean)

            return threshold

        except Exception as e:
            logger.debug(f"GMM threshold calculation failed: {e}")
            return None

    def _combine_threshold_methods(self, method_results: Dict[str, float],
                                 condition_data: pd.DataFrame) -> Tuple[float, float]:
        """Combine multiple threshold calculation methods with confidence weighting"""
        # Method reliability weights based on statistical rigor
        method_weights = {
            'roc': 0.3,        # High weight for ROC-based
            'pr_curve': 0.3,   # High weight for PR-based
            'percentile': 0.2, # Medium weight for percentile
            'gmm': 0.2         # Medium weight for GMM
        }

        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0

        for method, threshold in method_results.items():
            weight = method_weights.get(method, 0.1)
            weighted_sum += threshold * weight
            total_weight += weight

        if total_weight == 0:
            # Fallback to simple average
            combined_threshold = np.mean(list(method_results.values()))
            confidence = 0.5
        else:
            combined_threshold = weighted_sum / total_weight

            # Calculate confidence based on method agreement
            thresholds = list(method_results.values())
            relative_std = np.std(thresholds) / np.mean(thresholds) if np.mean(thresholds) > 0 else 1
            confidence = max(0.3, min(1.0, 1.0 - relative_std))

            # Boost confidence if we have more methods
            method_count_bonus = min(0.2, len(method_results) * 0.05)
            confidence = min(1.0, confidence + method_count_bonus)

            # Adjust confidence based on sample size
            sample_count = len(condition_data)
            if sample_count < 50:
                confidence *= 0.8
            elif sample_count > 200:
                confidence = min(1.0, confidence * 1.1)

        return combined_threshold, confidence

    def update_thresholds(self, force_update: bool = False,
                         specific_condition: str = None) -> Dict[str, Any]:
        """Update thresholds based on recent data with smart smoothing"""
        if not force_update and not self.should_update_thresholds():
            return {"status": "no_update_needed", "message": "Update frequency not reached"}

        logger.info("Starting adaptive threshold update...")

        # Calculate new optimal thresholds
        new_thresholds = self.calculate_optimal_thresholds(condition=specific_condition)

        if not new_thresholds:
            return {"status": "no_data", "message": "Insufficient data for threshold calculation"}

        # Update thresholds with intelligent smoothing
        update_summary = {
            "status": "updated",
            "updated_conditions": [],
            "threshold_changes": {},
            "timestamp": datetime.now().isoformat()
        }

        for condition, new_values in new_thresholds.items():
            if condition not in self.thresholds:
                # New condition - add with conservative confidence
                self.thresholds[condition] = {
                    'threshold': new_values['threshold'],
                    'confidence': max(0.5, new_values['confidence']),
                    'last_updated': datetime.now().isoformat(),
                    'update_count': 1,
                    'performance_history': []
                }
                update_summary["updated_conditions"].append(condition)
                update_summary["threshold_changes"][condition] = {
                    "old_threshold": None,
                    "new_threshold": new_values['threshold'],
                    "confidence": new_values['confidence'],
                    "change_reason": "new_condition"
                }
                continue

            # Existing condition - apply smart smoothing
            old_threshold = self.thresholds[condition]['threshold']
            old_confidence = self.thresholds[condition]['confidence']
            new_threshold = new_values['threshold']
            new_confidence = new_values['confidence']

            # Skip update if confidence is too low
            if new_confidence < self.confidence_threshold:
                logger.info(f"Skipping {condition} update due to low confidence: {new_confidence:.3f}")
                continue

            # Calculate relative change
            relative_change = abs(new_threshold - old_threshold) / old_threshold
            if relative_change > self.max_threshold_change:
                logger.warning(f"Large threshold change for {condition}: {relative_change:.3f}")
                # Cap the change
                direction = 1 if new_threshold > old_threshold else -1
                new_threshold = old_threshold * (1 + direction * self.max_threshold_change)

            # Determine smoothing factor based on confidence
            if new_confidence > 0.8:
                smoothing_factor = self.smoothing_factors['high_confidence']
            elif new_confidence > 0.6:
                smoothing_factor = self.smoothing_factors['medium_confidence']
            else:
                smoothing_factor = self.smoothing_factors['low_confidence']

            # Apply performance-based adjustment
            if len(self.thresholds[condition]['performance_history']) > 0:
                recent_performance = np.mean([
                    p['f1_score'] for p in self.thresholds[condition]['performance_history'][-5:]
                ])
                if recent_performance < 0.6:  # Poor recent performance
                    smoothing_factor *= 1.2  # Be more aggressive with updates
                elif recent_performance > 0.8:  # Good recent performance
                    smoothing_factor *= 0.8  # Be more conservative

            # Smooth update
            updated_threshold = (smoothing_factor * new_threshold +
                               (1 - smoothing_factor) * old_threshold)

            # Apply update if change is significant (>2% relative change)
            if abs(updated_threshold - old_threshold) / old_threshold > 0.02:
                self.thresholds[condition]['threshold'] = updated_threshold
                self.thresholds[condition]['confidence'] = new_confidence
                self.thresholds[condition]['last_updated'] = datetime.now().isoformat()
                self.thresholds[condition]['update_count'] += 1

                update_summary["updated_conditions"].append(condition)
                update_summary["threshold_changes"][condition] = {
                    "old_threshold": old_threshold,
                    "new_threshold": updated_threshold,
                    "raw_new_threshold": new_threshold,
                    "confidence": new_confidence,
                    "smoothing_factor": smoothing_factor,
                    "relative_change": abs(updated_threshold - old_threshold) / old_threshold,
                    "methods_used": new_values['methods_used']
                }

        # Update tracking variables
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                self.last_update_count = cursor.fetchone()[0]
        except:
            pass

        self.last_update_time = datetime.now()

        # Evaluate and store performance
        if update_summary["updated_conditions"]:
            performance = self.evaluate_current_performance()
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'thresholds': {k: v['threshold'] for k, v in self.thresholds.items()},
                'performance': performance,
                'update_summary': update_summary
            })

        logger.info(f"Threshold update complete. Updated {len(update_summary['updated_conditions'])} conditions")

        return update_summary

    def evaluate_current_performance(self, lookback_hours: int = 24) -> Dict[str, Dict[str, float]]:
        """Evaluate current threshold performance on recent data"""
        try:
            lookback_time = datetime.now() - timedelta(hours=lookback_hours)

            with sqlite3.connect(self.sqlite_db_path) as conn:
                query = """
                    SELECT error_score, anomaly_status, anomaly_type
                    FROM chunks
                    WHERE processing_timestamp >= ?
                    ORDER BY processing_timestamp DESC
                    LIMIT 1000
                """
                df = pd.read_sql_query(query, conn, params=(lookback_time.isoformat(),))

        except Exception as e:
            logger.error(f"Error loading performance evaluation data: {e}")
            return {}

        if len(df) == 0:
            return {}

        df['condition_clean'] = df['anomaly_type'].fillna('Unknown')
        performance = {}

        for condition in df['condition_clean'].unique():
            condition_data = df[df['condition_clean'] == condition]

            if len(condition_data) < 5:
                continue

            threshold = self.thresholds.get(condition, {}).get('threshold', 0.1)

            # Predict using current threshold
            y_true = (condition_data['anomaly_status'] == 'anomaly').astype(int)
            y_pred = (condition_data['error_score'] > threshold).astype(int)

            # Calculate metrics
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(condition_data)

            # Calculate additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

            performance[condition] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'specificity': round(specificity, 4),
                'false_positive_rate': round(false_positive_rate, 4),
                'sample_count': len(condition_data),
                'threshold_used': round(threshold, 4),
                'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
            }

            # Store in condition's performance history
            if condition in self.thresholds:
                if 'performance_history' not in self.thresholds[condition]:
                    self.thresholds[condition]['performance_history'] = []

                self.thresholds[condition]['performance_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy
                })

                # Keep only last 20 performance records
                self.thresholds[condition]['performance_history'] = \
                    self.thresholds[condition]['performance_history'][-20:]

        return performance

    def get_threshold_for_condition(self, condition: str) -> float:
        """Get current threshold for a specific condition"""
        return self.thresholds.get(
            condition,
            self.thresholds.get('Unknown', {'threshold': 0.1})
        )['threshold']

    def get_all_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get all current thresholds with metadata"""
        return self.thresholds.copy()

    def get_performance_history(self, condition: str = None) -> List[Dict]:
        """Get historical performance data"""
        if condition and condition in self.thresholds:
            return self.thresholds[condition].get('performance_history', [])
        return self.performance_history.copy()

    def reset_thresholds(self, condition: str = None):
        """Reset thresholds to initial values"""
        if condition:
            if condition in self.thresholds:
                self.load_initial_thresholds()
                logger.info(f"Reset threshold for {condition}")
        else:
            self.load_initial_thresholds()
            logger.info("Reset all thresholds to initial values")

    def export_threshold_config(self, filepath: str):
        """Export current threshold configuration to JSON file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'thresholds': self.thresholds,
                'performance_history': self.performance_history[-10:],  # Last 10 records
                'config': {
                    'update_frequency': self.update_frequency,
                    'confidence_threshold': self.confidence_threshold,
                    'smoothing_factors': self.smoothing_factors
                }
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Threshold configuration exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting threshold config: {e}")

    def import_threshold_config(self, filepath: str):
        """Import threshold configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)

            self.thresholds = import_data['thresholds']
            if 'performance_history' in import_data:
                self.performance_history.extend(import_data['performance_history'])

            logger.info(f"Threshold configuration imported from {filepath}")

        except Exception as e:
            logger.error(f"Error importing threshold config: {e}")

def run_threshold_optimization(sqlite_db_path: str = "rmsai_metadata.db",
                             output_file: str = "threshold_optimization_results.json") -> Dict[str, Any]:
    """Run comprehensive threshold optimization analysis"""
    logger.info("Starting threshold optimization analysis...")

    threshold_manager = AdaptiveThresholdManager(sqlite_db_path)

    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_results': {}
    }

    try:
        # 1. Calculate optimal thresholds
        logger.info("Calculating optimal thresholds...")
        optimal_thresholds = threshold_manager.calculate_optimal_thresholds()
        results['analysis_results']['optimal_thresholds'] = optimal_thresholds

        # 2. Update thresholds
        logger.info("Updating thresholds...")
        update_result = threshold_manager.update_thresholds(force_update=True)
        results['analysis_results']['update_result'] = update_result

        # 3. Evaluate performance
        logger.info("Evaluating performance...")
        performance = threshold_manager.evaluate_current_performance()
        results['analysis_results']['performance_evaluation'] = performance

        # 4. Get threshold history
        history = threshold_manager.get_performance_history()
        results['analysis_results']['performance_history'] = history[-5:]  # Last 5 records

        # 5. Export results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Threshold optimization complete. Results saved to {output_file}")

        # Print summary
        print("\nThreshold Optimization Summary:")
        print("=" * 50)

        if optimal_thresholds:
            for condition, data in optimal_thresholds.items():
                print(f"{condition}: {data['threshold']:.4f} (confidence: {data['confidence']:.3f})")

        if performance:
            avg_f1 = np.mean([p['f1_score'] for p in performance.values()])
            print(f"\nAverage F1 Score: {avg_f1:.3f}")

    except Exception as e:
        logger.error(f"Error in threshold optimization: {e}")
        results['error'] = str(e)

    return results

if __name__ == "__main__":
    # Example usage
    results = run_threshold_optimization()

    if 'error' not in results:
        print("Threshold optimization completed successfully!")
    else:
        print(f"Optimization failed: {results['error']}")