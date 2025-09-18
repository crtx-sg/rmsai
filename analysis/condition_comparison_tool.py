#!/usr/bin/env python3
"""
RMSAI Condition Comparison Tool
===============================

Compares input (ground truth) conditions vs predicted conditions from the autoencoder.
Provides detailed analysis including accuracy metrics, confusion matrix, and per-condition performance.

Usage:
    python condition_comparison_tool.py [--format table|csv|json] [--output filename] [--detailed]
"""

import sqlite3
import pandas as pd
import json
import argparse
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

class ConditionComparator:
    """Tool for comparing input conditions vs autoencoder predictions"""

    def __init__(self, db_path="rmsai_metadata.db"):
        self.db_path = db_path
        self.comparison_data = None
        self.metrics = None

    def load_comparison_data(self):
        """Load and prepare comparison data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT
                    chunk_id,
                    event_id,
                    source_file,
                    lead_name,
                    JSON_EXTRACT(metadata, '$.condition') as input_condition,
                    anomaly_type as predicted_condition,
                    anomaly_status,
                    error_score,
                    processing_timestamp
                FROM chunks
                ORDER BY event_id, chunk_id
                """

                df = pd.read_sql_query(query, conn)

                # Clean up conditions - handle nulls and normalize names
                df['input_condition'] = df['input_condition'].fillna('Unknown')
                df['predicted_condition'] = df['predicted_condition'].fillna('Unknown')

                # Map "Normal" in predicted to actual condition name for better comparison
                # This handles cases where autoencoder predicts "Normal" but should predict specific condition
                df['predicted_normalized'] = df['predicted_condition']

                self.comparison_data = df
                return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def calculate_accuracy_metrics(self):
        """Calculate accuracy metrics and confusion matrix"""
        if self.comparison_data is None:
            self.load_comparison_data()

        df = self.comparison_data

        # Basic accuracy calculation
        correct_predictions = (df['input_condition'] == df['predicted_condition']).sum()
        total_predictions = len(df)
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Per-condition accuracy
        condition_metrics = {}
        for condition in df['input_condition'].unique():
            condition_df = df[df['input_condition'] == condition]
            condition_correct = (condition_df['input_condition'] == condition_df['predicted_condition']).sum()
            condition_total = len(condition_df)
            condition_accuracy = condition_correct / condition_total if condition_total > 0 else 0

            condition_metrics[condition] = {
                'total_samples': condition_total,
                'correct_predictions': condition_correct,
                'accuracy': condition_accuracy,
                'avg_error_score': condition_df['error_score'].mean()
            }

        # Confusion matrix
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            confusion_matrix[row['input_condition']][row['predicted_condition']] += 1

        # Lead-wise performance
        lead_metrics = {}
        for lead in df['lead_name'].unique():
            lead_df = df[df['lead_name'] == lead]
            lead_correct = (lead_df['input_condition'] == lead_df['predicted_condition']).sum()
            lead_total = len(lead_df)
            lead_accuracy = lead_correct / lead_total if lead_total > 0 else 0

            lead_metrics[lead] = {
                'total_samples': lead_total,
                'correct_predictions': lead_correct,
                'accuracy': lead_accuracy,
                'avg_error_score': lead_df['error_score'].mean()
            }

        self.metrics = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_predictions,
            'correct_predictions': correct_predictions,
            'condition_metrics': condition_metrics,
            'lead_metrics': lead_metrics,
            'confusion_matrix': dict(confusion_matrix),
            'timestamp': datetime.now().isoformat()
        }

        return self.metrics

    def generate_comparison_table(self, limit=None, detailed=False):
        """Generate comparison table in various formats"""
        if self.comparison_data is None:
            self.load_comparison_data()

        df = self.comparison_data.copy()

        if limit:
            df = df.head(limit)

        # Create comparison result column
        df['prediction_result'] = df.apply(
            lambda row: '✓ Correct' if row['input_condition'] == row['predicted_condition'] else '✗ Incorrect',
            axis=1
        )

        # Select columns for output
        if detailed:
            columns = [
                'event_id', 'chunk_id', 'lead_name', 'input_condition',
                'predicted_condition', 'prediction_result', 'error_score',
                'anomaly_status', 'processing_timestamp'
            ]
        else:
            columns = [
                'event_id', 'input_condition', 'predicted_condition',
                'prediction_result', 'error_score'
            ]

        return df[columns]

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if self.metrics is None:
            self.calculate_accuracy_metrics()

        report = {
            'summary': {
                'overall_accuracy': f"{self.metrics['overall_accuracy']:.1%}",
                'total_samples': self.metrics['total_samples'],
                'correct_predictions': self.metrics['correct_predictions'],
                'incorrect_predictions': self.metrics['total_samples'] - self.metrics['correct_predictions']
            },
            'condition_performance': {},
            'lead_performance': {},
            'confusion_matrix': self.metrics['confusion_matrix']
        }

        # Format condition metrics
        for condition, metrics in self.metrics['condition_metrics'].items():
            report['condition_performance'][condition] = {
                'accuracy': f"{metrics['accuracy']:.1%}",
                'samples': metrics['total_samples'],
                'avg_error_score': f"{metrics['avg_error_score']:.4f}"
            }

        # Format lead metrics
        for lead, metrics in self.metrics['lead_metrics'].items():
            report['lead_performance'][lead] = {
                'accuracy': f"{metrics['accuracy']:.1%}",
                'samples': metrics['total_samples'],
                'avg_error_score': f"{metrics['avg_error_score']:.4f}"
            }

        return report

    def export_results(self, format='table', filename=None, detailed=False, limit=None):
        """Export results in specified format"""

        if format == 'table':
            df = self.generate_comparison_table(limit=limit, detailed=detailed)
            output = df.to_string(index=False)

        elif format == 'csv':
            df = self.generate_comparison_table(limit=limit, detailed=detailed)
            if filename:
                df.to_csv(filename, index=False)
                return f"Results exported to {filename}"
            else:
                output = df.to_csv(index=False)

        elif format == 'json':
            report = self.generate_summary_report()
            if filename:
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
                return f"Results exported to {filename}"
            else:
                output = json.dumps(report, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

        if filename and format == 'table':
            with open(filename, 'w') as f:
                f.write(output)
            return f"Results exported to {filename}"

        return output

def main():
    parser = argparse.ArgumentParser(description='Compare input vs predicted conditions')
    parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--detailed', action='store_true',
                        help='Include detailed columns')
    parser.add_argument('--limit', type=int, help='Limit number of rows')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary report instead of detailed table')

    args = parser.parse_args()

    # Create comparator
    comparator = ConditionComparator()

    try:
        # Load and calculate metrics
        print("Loading comparison data...")
        comparator.load_comparison_data()
        comparator.calculate_accuracy_metrics()

        if args.summary:
            # Show summary report
            report = comparator.generate_summary_report()

            print("\n" + "="*60)
            print("CONDITION COMPARISON SUMMARY REPORT")
            print("="*60)

            print(f"\nOverall Performance:")
            print(f"  Accuracy: {report['summary']['overall_accuracy']}")
            print(f"  Total Samples: {report['summary']['total_samples']:,}")
            print(f"  Correct Predictions: {report['summary']['correct_predictions']:,}")
            print(f"  Incorrect Predictions: {report['summary']['incorrect_predictions']:,}")

            print(f"\nPer-Condition Performance:")
            for condition, metrics in report['condition_performance'].items():
                print(f"  {condition}:")
                print(f"    Accuracy: {metrics['accuracy']}")
                print(f"    Samples: {metrics['samples']:,}")
                print(f"    Avg Error Score: {metrics['avg_error_score']}")

            print(f"\nPer-Lead Performance:")
            for lead, metrics in report['lead_performance'].items():
                print(f"  {lead}:")
                print(f"    Accuracy: {metrics['accuracy']}")
                print(f"    Samples: {metrics['samples']:,}")

            print(f"\nConfusion Matrix:")
            print(f"{'Input/Predicted':<20}", end='')
            all_conditions = set()
            for input_cond in report['confusion_matrix']:
                for pred_cond in report['confusion_matrix'][input_cond]:
                    all_conditions.add(pred_cond)

            for cond in sorted(all_conditions):
                print(f"{cond:<15}", end='')
            print()

            for input_cond in sorted(report['confusion_matrix'].keys()):
                print(f"{input_cond:<20}", end='')
                for pred_cond in sorted(all_conditions):
                    count = report['confusion_matrix'].get(input_cond, {}).get(pred_cond, 0)
                    print(f"{count:<15}", end='')
                print()

        else:
            # Generate and display/export comparison table
            result = comparator.export_results(
                format=args.format,
                filename=args.output,
                detailed=args.detailed,
                limit=args.limit
            )

            if args.output:
                print(result)
            else:
                print("\n" + "="*80)
                print("CONDITION COMPARISON RESULTS")
                print("="*80)
                print(result)

                # Also show quick summary
                metrics = comparator.metrics
                print(f"\nQuick Summary:")
                print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%} ({metrics['correct_predictions']}/{metrics['total_samples']})")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())