#!/usr/bin/env python3
"""
Simple Test Suite for ECG Time Series Anomaly Detection

This is a clean, straightforward test suite for the ECG anomaly detection system.
No configuration files, no complex setup - just simple, readable tests.
"""

import unittest
import torch
import numpy as np
import pandas as pd
import warnings
import sys
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up matplotlib to not require display
import matplotlib
matplotlib.use('Agg')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Use CPU for consistent, simple testing
device = torch.device('cpu')

# Simple, clear model implementations
class Encoder(torch.nn.Module):
    def __init__(self, seq_len=140, n_features=1, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = torch.nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(torch.nn.Module):
    def __init__(self, seq_len=140, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_features = n_features
        self.hidden_dim = 2 * input_dim

        self.rnn1 = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class RecurrentAutoencoder(torch.nn.Module):
    def __init__(self, seq_len=140, n_features=1, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Simple utility functions
def create_dataset(df):
    """Convert DataFrame to tensor dataset"""
    sequences = df.astype(np.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    
    if len(dataset) > 0:
        n_seq, seq_len, n_features = torch.stack(dataset).shape
    else:
        seq_len, n_features = 140, 1
    
    return dataset, seq_len, n_features


def predict(model, dataset):
    """Make predictions using the model"""
    predictions, losses = [], []
    criterion = torch.nn.L1Loss(reduction='sum')
    
    with torch.no_grad():
        model.eval()
        for seq_true in dataset:
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    
    return predictions, losses


def train_model(model, train_dataset, val_dataset, n_epochs=20):
    """Train the model - simplified version for testing"""
    import copy
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.L1Loss(reduction='sum')
    history = {'train': [], 'val': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        train_losses = []
        for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def evaluate_anomaly_detection(normal_losses, anomaly_losses, threshold):
    """Calculate accuracy, precision, recall, F1-score"""
    # Make predictions: 0 = normal, 1 = anomaly
    normal_predictions = [0 if loss <= threshold else 1 for loss in normal_losses]
    anomaly_predictions = [0 if loss <= threshold else 1 for loss in anomaly_losses]
    
    # True labels
    normal_labels = [0] * len(normal_losses)
    anomaly_labels = [1] * len(anomaly_losses)
    
    # Combine for metrics calculation
    all_predictions = normal_predictions + anomaly_predictions
    all_labels = normal_labels + anomaly_labels
    
    # Calculate confusion matrix
    tp = sum(1 for pred, label in zip(all_predictions, all_labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(all_predictions, all_labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(all_predictions, all_labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(all_predictions, all_labels) if pred == 0 and label == 1)
    
    # Calculate metrics
    total = len(all_predictions)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }


# Test Classes
class TestEncoder(unittest.TestCase):
    """Test the Encoder class"""
    
    def test_encoder_initialization(self):
        encoder = Encoder(seq_len=140, n_features=1, embedding_dim=64)
        self.assertEqual(encoder.seq_len, 140)
        self.assertEqual(encoder.n_features, 1)
        self.assertEqual(encoder.embedding_dim, 64)
    
    def test_encoder_forward_pass(self):
        encoder = Encoder(seq_len=140, n_features=1, embedding_dim=64)
        input_tensor = torch.randn(140, 1)
        output = encoder(input_tensor)
        self.assertEqual(output.shape, (1, 64))
    
    def test_encoder_different_sizes(self):
        for embedding_dim in [32, 64, 128]:
            encoder = Encoder(seq_len=140, n_features=1, embedding_dim=embedding_dim)
            input_tensor = torch.randn(140, 1)
            output = encoder(input_tensor)
            self.assertEqual(output.shape, (1, embedding_dim))


class TestDecoder(unittest.TestCase):
    """Test the Decoder class"""
    
    def test_decoder_initialization(self):
        decoder = Decoder(seq_len=140, input_dim=64, n_features=1)
        self.assertEqual(decoder.seq_len, 140)
        self.assertEqual(decoder.input_dim, 64)
        self.assertEqual(decoder.n_features, 1)
    
    def test_decoder_forward_pass(self):
        decoder = Decoder(seq_len=140, input_dim=64, n_features=1)
        input_tensor = torch.randn(1, 64)
        output = decoder(input_tensor)
        self.assertEqual(output.shape, (140, 1))
    
    def test_decoder_different_sizes(self):
        for input_dim in [32, 64, 128]:
            decoder = Decoder(seq_len=140, input_dim=input_dim, n_features=1)
            input_tensor = torch.randn(1, input_dim)
            output = decoder(input_tensor)
            self.assertEqual(output.shape, (140, 1))


class TestRecurrentAutoencoder(unittest.TestCase):
    """Test the complete autoencoder"""
    
    def test_autoencoder_initialization(self):
        model = RecurrentAutoencoder(seq_len=140, n_features=1, embedding_dim=64)
        self.assertIsInstance(model.encoder, Encoder)
        self.assertIsInstance(model.decoder, Decoder)
    
    def test_autoencoder_forward_pass(self):
        model = RecurrentAutoencoder(seq_len=140, n_features=1, embedding_dim=64)
        input_tensor = torch.randn(140, 1)
        output = model(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)
    
    def test_autoencoder_loss_calculation(self):
        model = RecurrentAutoencoder(seq_len=140, n_features=1, embedding_dim=64)
        input_tensor = torch.randn(140, 1)
        output = model(input_tensor)
        
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(output, input_tensor)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_dataset(self):
        # Create test data
        data = np.random.randn(10, 140)
        df = pd.DataFrame(data)
        
        dataset, seq_len, n_features = create_dataset(df)
        
        self.assertEqual(len(dataset), 10)
        self.assertEqual(seq_len, 140)
        self.assertEqual(n_features, 1)
        
        for tensor in dataset:
            self.assertEqual(tensor.shape, (140, 1))
            self.assertEqual(tensor.dtype, torch.float32)
    
    def test_create_dataset_empty(self):
        df = pd.DataFrame()
        dataset, seq_len, n_features = create_dataset(df)
        
        self.assertEqual(len(dataset), 0)
        self.assertEqual(seq_len, 140)
        self.assertEqual(n_features, 1)
    
    def test_predict_function(self):
        model = RecurrentAutoencoder(seq_len=10, n_features=1, embedding_dim=16)
        dataset = [torch.randn(10, 1) for _ in range(3)]
        
        predictions, losses = predict(model, dataset)
        
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))
        self.assertTrue(all(loss >= 0 for loss in losses))


class TestModelTraining(unittest.TestCase):
    """Test model training"""
    
    def test_train_model_basic(self):
        train_dataset = [torch.randn(10, 1) for _ in range(5)]
        val_dataset = [torch.randn(10, 1) for _ in range(2)]
        
        model = RecurrentAutoencoder(seq_len=10, n_features=1, embedding_dim=16)
        trained_model, history = train_model(model, train_dataset, val_dataset, n_epochs=3)
        
        self.assertIn('train', history)
        self.assertIn('val', history)
        self.assertEqual(len(history['train']), 3)
        self.assertEqual(len(history['val']), 3)
        
        # Check all losses are positive
        self.assertTrue(all(loss >= 0 for loss in history['train']))
        self.assertTrue(all(loss >= 0 for loss in history['val']))


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics"""
    
    def test_perfect_classification(self):
        normal_losses = [1.0, 1.5, 2.0, 1.2, 1.8]
        anomaly_losses = [5.0, 6.5, 7.0, 5.5, 8.0]
        threshold = 3.0
        
        metrics = evaluate_anomaly_detection(normal_losses, anomaly_losses, threshold)
        
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
        self.assertEqual(metrics['true_positives'], 5)
        self.assertEqual(metrics['true_negatives'], 5)
        self.assertEqual(metrics['false_positives'], 0)
        self.assertEqual(metrics['false_negatives'], 0)
    
    def test_mixed_classification(self):
        normal_losses = [1.0, 2.0, 4.0, 1.5]  # One false positive
        anomaly_losses = [2.5, 5.0, 6.0, 7.0]  # One false negative
        threshold = 3.0
        
        metrics = evaluate_anomaly_detection(normal_losses, anomaly_losses, threshold)
        
        expected_accuracy = 6/8  # 3 TP + 3 TN out of 8 total
        self.assertEqual(metrics['accuracy'], expected_accuracy)
        self.assertEqual(metrics['precision'], 3/4)
        self.assertEqual(metrics['recall'], 3/4)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_complete_pipeline(self):
        # Create synthetic data
        normal_data = np.random.randn(20, 50) * 0.5  # Small variance
        anomaly_data = np.random.randn(10, 50) * 2.0  # Large variance
        
        # Create datasets
        normal_df = pd.DataFrame(normal_data)
        anomaly_df = pd.DataFrame(anomaly_data)
        
        train_dataset, seq_len, n_features = create_dataset(normal_df[:15])
        val_dataset, _, _ = create_dataset(normal_df[15:])
        test_normal_dataset, _, _ = create_dataset(normal_df[18:])
        test_anomaly_dataset, _, _ = create_dataset(anomaly_df[:5])
        
        # Train model
        model = RecurrentAutoencoder(seq_len, n_features, 32)
        trained_model, history = train_model(model, train_dataset, val_dataset, n_epochs=5)
        
        # Test predictions
        _, normal_losses = predict(trained_model, test_normal_dataset)
        _, anomaly_losses = predict(trained_model, test_anomaly_dataset)
        
        # Evaluate
        threshold = np.mean(normal_losses) + 2 * np.std(normal_losses)
        metrics = evaluate_anomaly_detection(normal_losses, anomaly_losses, threshold)
        
        # Basic sanity checks
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(0 <= metrics['recall'] <= 1)
        self.assertTrue(0 <= metrics['f1_score'] <= 1)
        
        # Model should generally do better than random
        self.assertGreater(metrics['accuracy'], 0.3)


def run_tests():
    """Run all tests with simple output"""
    print("=" * 60)
    print("ECG Anomaly Detection - Simple Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    test_classes = [
        TestEncoder,
        TestDecoder,
        TestRecurrentAutoencoder,
        TestUtilityFunctions,
        TestModelTraining,
        TestEvaluationMetrics,
        TestEndToEnd
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Simple summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {result.testsRun} tests run")
    print(f"✓ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ Failed: {len(result.failures)}")
    print(f"✗ Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed - check output above")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)