#!/usr/bin/env python3
"""
RMSAI Advanced Analytics Module
===============================

Machine learning analytics applied to ECG embeddings for pattern discovery,
clustering, anomaly detection, and clinical insights.

Features:
- Embedding space clustering (K-means, DBSCAN)
- Anomalous pattern detection (Isolation Forest, One-class SVM, LOF)
- Temporal pattern analysis
- Similarity network generation
- 2D visualization (PCA, UMAP)
- Clinical correlation analysis

Usage:
    from advanced_analytics import EmbeddingAnalytics

    analytics = EmbeddingAnalytics("vector_db", "rmsai_metadata.db")
    clusters = analytics.discover_embedding_clusters()
    patterns = analytics.detect_anomalous_patterns()
    analytics.visualize_embedding_space("embedding_viz.png")
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import sqlite3
import json
import logging
from datetime import datetime, timedelta

# Optional imports
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingAnalytics:
    """Advanced analytics on ECG embeddings for pattern discovery"""

    def __init__(self, vector_db_path: str, sqlite_db_path: str, selected_leads: Optional[List[str]] = None):
        self.vector_db_path = vector_db_path
        self.sqlite_db_path = sqlite_db_path
        self.selected_leads = selected_leads  # Filter analysis to specific leads if provided
        self.embeddings_cache = None
        self.metadata_cache = None
        self.cache_timestamp = None
        self.cache_duration = 300  # 5 minutes

        # Verify dependencies
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - vector operations will be limited")

        # Get available leads if none specified
        if self.selected_leads is None:
            self.selected_leads = self._get_available_leads()

        logger.info(f"Analytics configured for leads: {self.selected_leads}")

    def _get_available_leads(self) -> List[str]:
        """Get list of available leads from database"""
        try:
            with sqlite3.connect(self.sqlite_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT lead_name FROM chunks ORDER BY lead_name")
                leads = [row[0] for row in cursor.fetchall()]
                return leads if leads else ['ECG1', 'ECG2', 'ECG3', 'aVR', 'aVL', 'aVF', 'vVX']
        except Exception as e:
            logger.warning(f"Could not get available leads: {e}")
            return ['ECG1', 'ECG2', 'ECG3', 'aVR', 'aVL', 'aVF', 'vVX']

    def set_selected_leads(self, leads: List[str]):
        """Update the selected leads for analysis"""
        available_leads = self._get_available_leads()
        invalid_leads = [lead for lead in leads if lead not in available_leads]
        if invalid_leads:
            logger.warning(f"Invalid leads specified: {invalid_leads}. Available: {available_leads}")
            leads = [lead for lead in leads if lead in available_leads]

        self.selected_leads = leads
        # Clear cache to force reload with new lead filter
        self.embeddings_cache = None
        self.metadata_cache = None
        self.cache_timestamp = None
        logger.info(f"Updated analytics to focus on leads: {self.selected_leads}")

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if (self.embeddings_cache is None or self.metadata_cache is None or
            self.cache_timestamp is None):
            return False

        elapsed = (datetime.now() - self.cache_timestamp).total_seconds()
        return elapsed < self.cache_duration

    def load_embeddings_with_metadata(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load all embeddings with associated metadata"""
        if self._is_cache_valid():
            return self.embeddings_cache, self.metadata_cache

        logger.info("Loading embeddings and metadata...")

        if not CHROMADB_AVAILABLE:
            raise RuntimeError("ChromaDB not available for embedding operations")

        try:
            # Load from ChromaDB
            client = chromadb.PersistentClient(path=self.vector_db_path)
            collection = client.get_collection("rmsai_ecg_embeddings")

            # Get all embeddings
            results = collection.get(include=['embeddings', 'metadatas'])

            if results['embeddings'] is None or len(results['embeddings']) == 0:
                logger.warning("No embeddings found in vector database")
                return np.array([]), pd.DataFrame()

            embeddings = np.array(results['embeddings'])

            # Create metadata DataFrame
            metadata_df = pd.DataFrame(results['metadatas'])
            metadata_df['chunk_id'] = results['ids']

            # Enrich with SQL metadata
            with sqlite3.connect(self.sqlite_db_path) as conn:
                sql_data = pd.read_sql_query("""
                    SELECT chunk_id, error_score, anomaly_status,
                           processing_timestamp, lead_name, event_id, source_file,
                           anomaly_type
                    FROM chunks
                """, conn)

            # Merge datasets
            full_metadata = metadata_df.merge(sql_data, on='chunk_id', how='left')

            # Filter to selected leads if specified
            if self.selected_leads:
                before_filter = len(full_metadata)
                lead_mask = full_metadata['lead_name'].isin(self.selected_leads)
                full_metadata = full_metadata[lead_mask]
                embeddings = embeddings[lead_mask.values]
                after_filter = len(full_metadata)
                logger.info(f"Filtered data: {before_filter} -> {after_filter} chunks (leads: {self.selected_leads})")

            # Cache the results
            self.embeddings_cache = embeddings
            self.metadata_cache = full_metadata
            self.cache_timestamp = datetime.now()

            logger.info(f"Loaded {len(embeddings)} embeddings with metadata")
            return embeddings, full_metadata

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def discover_embedding_clusters(self, n_clusters: int = None,
                                  methods: List[str] = None) -> Dict[str, Any]:
        """Discover natural clusters in embedding space using multiple methods"""
        if methods is None:
            methods = ['kmeans', 'dbscan']

        embeddings, metadata = self.load_embeddings_with_metadata()

        if len(embeddings) == 0:
            return {"error": "No embeddings available for clustering"}

        logger.info(f"Clustering {len(embeddings)} embeddings using methods: {methods}")

        results = {}

        # DBSCAN clustering (density-based)
        if 'dbscan' in methods:
            try:
                # Adaptive epsilon based on embedding dimensionality
                eps = 0.5 if embeddings.shape[1] <= 64 else 0.7
                min_samples = max(5, int(len(embeddings) * 0.01))  # 1% of data, min 5

                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(embeddings)

                n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                noise_points = np.sum(dbscan_labels == -1)

                results['dbscan'] = {
                    'labels': dbscan_labels.tolist(),
                    'n_clusters': n_clusters_dbscan,
                    'noise_points': int(noise_points),
                    'parameters': {'eps': eps, 'min_samples': min_samples}
                }

                logger.info(f"DBSCAN found {n_clusters_dbscan} clusters with {noise_points} noise points")

            except Exception as e:
                logger.error(f"DBSCAN clustering failed: {e}")
                results['dbscan'] = {'error': str(e)}

        # K-means clustering
        if 'kmeans' in methods:
            try:
                if n_clusters is None:
                    # Find optimal number of clusters using silhouette analysis
                    max_k = min(15, len(embeddings) // 10, 50)
                    if max_k < 2:
                        max_k = 2

                    k_range = range(2, max_k + 1)
                    silhouette_scores = []
                    inertias = []

                    for k in k_range:
                        if k > len(embeddings):
                            break

                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(embeddings)

                        inertias.append(kmeans.inertia_)

                        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
                            silhouette_scores.append(silhouette_score(embeddings, labels))
                        else:
                            silhouette_scores.append(0)

                    # Choose k with best silhouette score
                    if silhouette_scores:
                        optimal_k = k_range[np.argmax(silhouette_scores)]
                        max_silhouette = max(silhouette_scores)
                    else:
                        optimal_k = 2
                        max_silhouette = 0
                else:
                    optimal_k = n_clusters
                    max_silhouette = 0
                    inertias = []

                # Fit final K-means model
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(embeddings)

                if len(set(kmeans_labels)) > 1:
                    final_silhouette = silhouette_score(embeddings, kmeans_labels)
                else:
                    final_silhouette = 0

                results['kmeans'] = {
                    'labels': kmeans_labels.tolist(),
                    'n_clusters': optimal_k,
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'silhouette_score': round(final_silhouette, 4),
                    'inertia': round(kmeans.inertia_, 4),
                    'optimization_scores': silhouette_scores if n_clusters is None else []
                }

                logger.info(f"K-means with k={optimal_k}, silhouette score: {final_silhouette:.4f}")

                # Analyze cluster characteristics
                results['cluster_analysis'] = self._analyze_clusters(
                    embeddings, metadata, kmeans_labels
                )

            except Exception as e:
                logger.error(f"K-means clustering failed: {e}")
                results['kmeans'] = {'error': str(e)}

        return results

    def _analyze_clusters(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                         labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of discovered clusters"""
        cluster_analysis = {}

        unique_labels = np.unique(labels)
        logger.info(f"Analyzing {len(unique_labels)} clusters")

        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_mask = labels == cluster_id
            cluster_metadata = metadata[cluster_mask].copy()
            cluster_embeddings = embeddings[cluster_mask]

            if len(cluster_metadata) == 0:
                continue

            # Clinical condition distribution
            if 'condition' in cluster_metadata.columns:
                condition_dist = cluster_metadata['condition'].value_counts()
            else:
                condition_dist = pd.Series(dtype='int64')

            # Lead distribution
            if 'lead_name' in cluster_metadata.columns:
                lead_dist = cluster_metadata['lead_name'].value_counts()
            else:
                lead_dist = pd.Series(dtype='int64')

            # Error score statistics
            if 'error_score' in cluster_metadata.columns:
                error_stats = cluster_metadata['error_score'].describe()
            else:
                error_stats = pd.Series([0, 0, 0, 0, 0, 0, 0, 0], index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

            # Anomaly rate
            if 'anomaly_status' in cluster_metadata.columns:
                anomaly_rate = (cluster_metadata['anomaly_status'] == 'anomaly').mean()
            else:
                anomaly_rate = 0.0

            # Embedding statistics
            centroid = np.mean(cluster_embeddings, axis=0)
            spread = np.std(cluster_embeddings, axis=0).mean()  # Average std across dimensions

            # Temporal distribution
            if 'processing_timestamp' in cluster_metadata.columns:
                cluster_metadata['timestamp'] = pd.to_datetime(cluster_metadata['processing_timestamp'])
                temporal_span = (cluster_metadata['timestamp'].max() -
                               cluster_metadata['timestamp'].min()).total_seconds() / 3600  # hours
            else:
                temporal_span = 0

            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': int(cluster_mask.sum()),
                'condition_distribution': condition_dist.to_dict(),
                'lead_distribution': lead_dist.to_dict(),
                'error_score_stats': {
                    'mean': round(error_stats['mean'], 4),
                    'std': round(error_stats['std'], 4),
                    'min': round(error_stats['min'], 4),
                    'max': round(error_stats['max'], 4)
                },
                'anomaly_rate': round(anomaly_rate, 4),
                'dominant_condition': condition_dist.index[0] if len(condition_dist) > 0 else None,
                'dominant_lead': lead_dist.index[0] if len(lead_dist) > 0 else None,
                'embedding_stats': {
                    'centroid_norm': round(np.linalg.norm(centroid), 4),
                    'spread': round(spread, 4),
                    'dimensions': len(centroid)
                },
                'temporal_span_hours': round(temporal_span, 2)
            }

        return cluster_analysis

    def detect_anomalous_patterns(self, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalous patterns using multiple unsupervised methods"""
        embeddings, metadata = self.load_embeddings_with_metadata()

        if len(embeddings) == 0:
            return {"error": "No embeddings available for anomaly detection"}

        logger.info(f"Detecting anomalous patterns in {len(embeddings)} embeddings")

        results = {}

        try:
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_scores = iso_forest.fit_predict(embeddings)
            iso_anomalies = np.where(iso_scores == -1)[0]

            results['isolation_forest'] = {
                'anomaly_indices': iso_anomalies.tolist(),
                'n_anomalies': len(iso_anomalies),
                'contamination': contamination
            }

        except Exception as e:
            logger.error(f"Isolation Forest failed: {e}")
            results['isolation_forest'] = {'error': str(e)}

        try:
            # One-class SVM
            svm = OneClassSVM(gamma='scale', nu=contamination)
            svm_scores = svm.fit_predict(embeddings)
            svm_anomalies = np.where(svm_scores == -1)[0]

            results['one_class_svm'] = {
                'anomaly_indices': svm_anomalies.tolist(),
                'n_anomalies': len(svm_anomalies),
                'nu': contamination
            }

        except Exception as e:
            logger.error(f"One-class SVM failed: {e}")
            results['one_class_svm'] = {'error': str(e)}

        try:
            # Local Outlier Factor
            lof = LocalOutlierFactor(n_neighbors=min(20, len(embeddings) // 2),
                                   contamination=contamination)
            lof_scores = lof.fit_predict(embeddings)
            lof_anomalies = np.where(lof_scores == -1)[0]

            results['local_outlier_factor'] = {
                'anomaly_indices': lof_anomalies.tolist(),
                'n_anomalies': len(lof_anomalies),
                'n_neighbors': min(20, len(embeddings) // 2)
            }

        except Exception as e:
            logger.error(f"LOF failed: {e}")
            results['local_outlier_factor'] = {'error': str(e)}

        # Find consensus anomalies (detected by multiple methods)
        all_methods = ['isolation_forest', 'one_class_svm', 'local_outlier_factor']
        available_methods = [m for m in all_methods if m in results and 'error' not in results[m]]

        if len(available_methods) >= 2:
            # Find intersection of anomalies from available methods
            anomaly_sets = [set(results[method]['anomaly_indices']) for method in available_methods]
            consensus_anomalies = set.intersection(*anomaly_sets)

            results['consensus'] = {
                'anomaly_indices': list(consensus_anomalies),
                'n_anomalies': len(consensus_anomalies),
                'methods_used': available_methods,
                'agreement_rate': len(consensus_anomalies) / max(len(s) for s in anomaly_sets) if anomaly_sets else 0
            }

            # Analyze consensus anomalies
            if consensus_anomalies:
                consensus_metadata = metadata.iloc[list(consensus_anomalies)]
                consensus_analysis = {}

                if 'condition' in consensus_metadata.columns:
                    consensus_analysis['condition_distribution'] = consensus_metadata['condition'].value_counts().to_dict()

                if 'lead_name' in consensus_metadata.columns:
                    consensus_analysis['lead_distribution'] = consensus_metadata['lead_name'].value_counts().to_dict()

                if 'error_score' in consensus_metadata.columns:
                    consensus_analysis['avg_error_score'] = round(consensus_metadata['error_score'].mean(), 4)

                if 'anomaly_status' in consensus_metadata.columns:
                    consensus_analysis['anomaly_rate'] = round((consensus_metadata['anomaly_status'] == 'anomaly').mean(), 4)

                results['consensus_analysis'] = consensus_analysis

        return results

    def temporal_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze temporal patterns in anomaly detection and embeddings"""
        embeddings, metadata = self.load_embeddings_with_metadata()

        if len(embeddings) == 0:
            return {"error": "No data available for temporal analysis"}

        logger.info("Analyzing temporal patterns...")

        # Convert timestamps
        metadata['timestamp'] = pd.to_datetime(metadata['processing_timestamp'])
        metadata['hour'] = metadata['timestamp'].dt.hour
        metadata['day_of_week'] = metadata['timestamp'].dt.dayofweek
        metadata['date'] = metadata['timestamp'].dt.date

        results = {}

        # Hourly patterns
        results['hourly_patterns'] = {}

        if 'anomaly_status' in metadata.columns:
            hourly_anomalies = metadata.groupby('hour')['anomaly_status'].apply(
                lambda x: (x == 'anomaly').mean()
            )
            results['hourly_patterns']['anomaly_rates'] = hourly_anomalies.to_dict()

        if 'error_score' in metadata.columns:
            hourly_error_scores = metadata.groupby('hour')['error_score'].mean()
            results['hourly_patterns']['avg_error_scores'] = hourly_error_scores.to_dict()

        # Daily patterns
        results['daily_patterns'] = {
            'day_names': ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                         'Friday', 'Saturday', 'Sunday']
        }

        if 'anomaly_status' in metadata.columns:
            daily_anomalies = metadata.groupby('day_of_week')['anomaly_status'].apply(
                lambda x: (x == 'anomaly').mean()
            )
            results['daily_patterns']['anomaly_rates'] = daily_anomalies.to_dict()

        # Condition temporal patterns
        results['condition_temporal'] = {
            'patterns': {},
            'peak_hours': {}
        }

        if 'condition' in metadata.columns:
            condition_temporal = metadata.groupby(['condition', 'hour']).size().unstack(fill_value=0)
            results['condition_temporal']['patterns'] = condition_temporal.to_dict()

            # Find peak hours for each condition
            for condition in condition_temporal.index:
                peak_hour = condition_temporal.loc[condition].idxmax()
                results['condition_temporal']['peak_hours'][condition] = int(peak_hour)

        # Processing volume over time
        daily_volume = metadata.groupby('date').size()

        results['volume_trends'] = {
            'daily_counts': {str(date): count for date, count in daily_volume.items()},
            'avg_daily_volume': round(daily_volume.mean(), 2),
            'peak_day': str(daily_volume.idxmax()) if len(daily_volume) > 0 else None,
            'total_days': len(daily_volume)
        }

        return results

    def generate_similarity_network(self, similarity_threshold: float = 0.8,
                                  max_connections: int = 1000) -> Dict[str, Any]:
        """Generate a network of similar ECG patterns"""
        embeddings, metadata = self.load_embeddings_with_metadata()

        if len(embeddings) == 0:
            return {"error": "No embeddings available for network generation"}

        logger.info(f"Generating similarity network with threshold {similarity_threshold}")

        # Calculate pairwise similarities (can be memory intensive)
        if len(embeddings) > 1000:
            logger.warning(f"Large number of embeddings ({len(embeddings)}), this may be slow")

        try:
            similarity_matrix = cosine_similarity(embeddings)

            # Create adjacency matrix
            adjacency = (similarity_matrix > similarity_threshold) & (similarity_matrix < 1.0)

            # Limit connections to prevent memory issues
            connection_count = np.sum(adjacency)
            if connection_count > max_connections:
                # Keep only the strongest connections
                threshold_adjustment = np.percentile(
                    similarity_matrix[adjacency],
                    100 * (1 - max_connections / connection_count)
                )
                adjacency = (similarity_matrix > threshold_adjustment) & (similarity_matrix < 1.0)
                logger.info(f"Adjusted threshold to {threshold_adjustment:.3f} to limit connections")

            # Find connected components (groups of similar patterns)
            sparse_adj = csr_matrix(adjacency)
            n_components, labels = connected_components(sparse_adj)

            # Analyze similarity groups
            similarity_groups = {}
            group_sizes = []

            for group_id in range(n_components):
                group_mask = labels == group_id
                group_size = group_mask.sum()

                if group_size > 1:  # Only groups with multiple members
                    group_metadata = metadata[group_mask]
                    group_embeddings = embeddings[group_mask]

                    # Calculate internal similarity
                    if group_size > 1:
                        group_similarities = similarity_matrix[np.ix_(group_mask, group_mask)]
                        avg_internal_similarity = np.mean(group_similarities[group_similarities < 1.0])
                    else:
                        avg_internal_similarity = 0

                    group_data = {
                        'size': int(group_size),
                        'avg_internal_similarity': round(avg_internal_similarity, 4)
                    }

                    if 'chunk_id' in group_metadata.columns:
                        group_data['chunk_ids'] = group_metadata['chunk_id'].tolist()[:50]  # Limit for output size

                    if 'condition' in group_metadata.columns:
                        group_data['conditions'] = group_metadata['condition'].value_counts().to_dict()

                    if 'lead_name' in group_metadata.columns:
                        group_data['leads'] = group_metadata['lead_name'].value_counts().to_dict()

                    if 'error_score' in group_metadata.columns:
                        group_data['avg_error_score'] = round(group_metadata['error_score'].mean(), 4)

                    if 'anomaly_status' in group_metadata.columns:
                        group_data['anomaly_rate'] = round((group_metadata['anomaly_status'] == 'anomaly').mean(), 4)

                    similarity_groups[f'group_{group_id}'] = group_data

                    group_sizes.append(group_size)

            # Network statistics
            total_connections = int(np.sum(adjacency) // 2)  # Undirected graph
            avg_degree = total_connections * 2 / len(embeddings) if len(embeddings) > 0 else 0

            results = {
                'n_similarity_groups': len(similarity_groups),
                'similarity_groups': similarity_groups,
                'network_stats': {
                    'total_nodes': len(embeddings),
                    'total_connections': total_connections,
                    'avg_degree': round(avg_degree, 2),
                    'largest_group_size': max(group_sizes) if group_sizes else 0,
                    'avg_group_size': round(np.mean(group_sizes), 2) if group_sizes else 0
                },
                'similarity_threshold': similarity_threshold,
                'max_connections_limit': max_connections
            }

            return results

        except Exception as e:
            logger.error(f"Error generating similarity network: {e}")
            return {"error": str(e)}

    def visualize_embedding_space(self, save_path: str = None,
                                 method: str = 'both') -> Dict[str, Any]:
        """Create 2D visualization of embedding space"""
        embeddings, metadata = self.load_embeddings_with_metadata()

        if len(embeddings) == 0:
            return {"error": "No embeddings available for visualization"}

        if not PLOTTING_AVAILABLE:
            return {"error": "Matplotlib/Seaborn not available for plotting"}

        logger.info(f"Creating {method} visualization of embedding space")

        # Reduce dimensionality for visualization
        reduced_embeddings = {}

        if method in ['pca', 'both']:
            try:
                pca = PCA(n_components=2, random_state=42)
                pca_embeddings = pca.fit_transform(embeddings)
                reduced_embeddings['pca'] = {
                    'embeddings': pca_embeddings,
                    'explained_variance': pca.explained_variance_ratio_.tolist()
                }
            except Exception as e:
                logger.error(f"PCA failed: {e}")

        if method in ['umap', 'both'] and UMAP_AVAILABLE:
            try:
                umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
                umap_embeddings = umap_reducer.fit_transform(embeddings)
                reduced_embeddings['umap'] = {
                    'embeddings': umap_embeddings
                }
            except Exception as e:
                logger.error(f"UMAP failed: {e}")

        if not reduced_embeddings:
            return {"error": "No dimensionality reduction methods available"}

        # Create visualization
        n_plots = len(reduced_embeddings) * 2  # 2 plots per method (condition + anomaly)
        n_cols = 2
        n_rows = (n_plots + 1) // 2

        plt.figure(figsize=(15, 6 * n_rows))
        plot_idx = 1

        for method_name, method_data in reduced_embeddings.items():
            coords = method_data['embeddings']

            # Plot by condition
            plt.subplot(n_rows, n_cols, plot_idx)
            unique_conditions = metadata['condition'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_conditions)))

            for i, condition in enumerate(unique_conditions):
                mask = metadata['condition'] == condition
                plt.scatter(coords[mask, 0], coords[mask, 1],
                          c=[colors[i]], label=condition, alpha=0.7, s=20)

            plt.title(f'{method_name.upper()} - By Condition')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            if method_name == 'pca':
                var_explained = method_data['explained_variance']
                plt.xlabel(f'PC1 ({var_explained[0]:.1%} variance)')
                plt.ylabel(f'PC2 ({var_explained[1]:.1%} variance)')

            plot_idx += 1

            # Plot by anomaly status
            plt.subplot(n_rows, n_cols, plot_idx)
            anomaly_mask = metadata['anomaly_status'] == 'anomaly'

            plt.scatter(coords[~anomaly_mask, 0], coords[~anomaly_mask, 1],
                       c='blue', label='Normal', alpha=0.7, s=20)
            plt.scatter(coords[anomaly_mask, 0], coords[anomaly_mask, 1],
                       c='red', label='Anomaly', alpha=0.8, s=25)

            plt.title(f'{method_name.upper()} - By Anomaly Status')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()

            if method_name == 'pca':
                var_explained = method_data['explained_variance']
                plt.xlabel(f'PC1 ({var_explained[0]:.1%} variance)')
                plt.ylabel(f'PC2 ({var_explained[1]:.1%} variance)')

            plot_idx += 1

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        plt.show()

        # Return analysis results
        analysis_results = {
            'methods_used': list(reduced_embeddings.keys()),
            'total_points': len(embeddings),
            'anomaly_separation': {},
            'condition_separation': {}
        }

        # Calculate separation metrics
        for method_name, method_data in reduced_embeddings.items():
            coords = method_data['embeddings']

            # Anomaly separation (distance between centroids)
            normal_centroid = np.mean(coords[~anomaly_mask], axis=0)
            anomaly_centroid = np.mean(coords[anomaly_mask], axis=0)
            separation_distance = np.linalg.norm(normal_centroid - anomaly_centroid)

            analysis_results['anomaly_separation'][method_name] = round(separation_distance, 4)

            # Condition separation (average pairwise centroid distances)
            condition_centroids = []
            for condition in unique_conditions:
                mask = metadata['condition'] == condition
                if np.any(mask):
                    centroid = np.mean(coords[mask], axis=0)
                    condition_centroids.append(centroid)

            if len(condition_centroids) > 1:
                pairwise_distances = []
                for i in range(len(condition_centroids)):
                    for j in range(i + 1, len(condition_centroids)):
                        dist = np.linalg.norm(condition_centroids[i] - condition_centroids[j])
                        pairwise_distances.append(dist)

                avg_condition_separation = np.mean(pairwise_distances)
                analysis_results['condition_separation'][method_name] = round(avg_condition_separation, 4)

        return analysis_results

    def analyze_pacer_patterns(self) -> Dict[str, Any]:
        """Analyze pacer information and timing patterns from HDF5 data"""
        logger.info("Analyzing pacer patterns...")

        # This method would need access to HDF5 files to read pacer data
        # For now, we'll analyze pacer patterns from metadata if available
        try:
            embeddings, metadata = self.load_embeddings_with_metadata()

            if len(embeddings) == 0:
                return {"error": "No embeddings available for pacer analysis"}

            # Check if pacer information is available in metadata
            pacer_columns = [col for col in metadata.columns if 'pacer' in col.lower()]

            if not pacer_columns:
                logger.warning("No pacer information found in metadata")
                return {"warning": "No pacer data available for analysis"}

            results = {
                'total_chunks_analyzed': len(metadata),
                'pacer_data_availability': {
                    'available_columns': pacer_columns,
                    'coverage_percentage': 100.0  # Assuming all new data has pacer info
                }
            }

            # If pacer type information is available
            if 'pacer_type' in metadata.columns:
                pacer_type_dist = metadata['pacer_type'].value_counts()
                results['pacer_type_distribution'] = {
                    'counts': pacer_type_dist.to_dict(),
                    'percentages': (pacer_type_dist / len(metadata) * 100).round(2).to_dict()
                }

                # Analyze pacer types by condition
                if 'condition' in metadata.columns:
                    pacer_by_condition = metadata.groupby('condition')['pacer_type'].value_counts()
                    results['pacer_by_condition'] = pacer_by_condition.to_dict()

            # If pacer offset/timing information is available
            if 'pacer_offset' in metadata.columns:
                pacer_offsets = metadata['pacer_offset'].dropna()

                if len(pacer_offsets) > 0:
                    # Convert sample offsets to time offsets (assuming 200 Hz ECG)
                    time_offsets = pacer_offsets / 200.0
                    window_percentages = (pacer_offsets / 2400.0) * 100

                    # Categorize timing
                    timing_categories = []
                    for percent in window_percentages:
                        if percent <= 25:
                            timing_categories.append("Early")
                        elif percent >= 75:
                            timing_categories.append("Late")
                        else:
                            timing_categories.append("Mid")

                    timing_dist = pd.Series(timing_categories).value_counts()

                    results['pacer_timing_analysis'] = {
                        'offset_statistics': {
                            'mean_samples': round(pacer_offsets.mean(), 2),
                            'std_samples': round(pacer_offsets.std(), 2),
                            'mean_seconds': round(time_offsets.mean(), 3),
                            'std_seconds': round(time_offsets.std(), 3),
                            'min_seconds': round(time_offsets.min(), 3),
                            'max_seconds': round(time_offsets.max(), 3)
                        },
                        'timing_distribution': timing_dist.to_dict(),
                        'window_position_stats': {
                            'mean_percent': round(window_percentages.mean(), 1),
                            'std_percent': round(window_percentages.std(), 1)
                        }
                    }

                    # Analyze timing by condition
                    if 'condition' in metadata.columns:
                        timing_by_condition = {}
                        for condition in metadata['condition'].unique():
                            condition_mask = metadata['condition'] == condition
                            condition_offsets = pacer_offsets[condition_mask]
                            if len(condition_offsets) > 0:
                                condition_time_offsets = condition_offsets / 200.0
                                timing_by_condition[condition] = {
                                    'count': len(condition_offsets),
                                    'mean_seconds': round(condition_time_offsets.mean(), 3),
                                    'std_seconds': round(condition_time_offsets.std(), 3)
                                }
                        results['timing_by_condition'] = timing_by_condition

            # Analyze error scores for events with different pacer configurations
            if 'error_score' in metadata.columns and 'pacer_type' in metadata.columns:
                pacer_error_analysis = {}
                for pacer_type in metadata['pacer_type'].unique():
                    pacer_mask = metadata['pacer_type'] == pacer_type
                    pacer_errors = metadata[pacer_mask]['error_score']
                    if len(pacer_errors) > 0:
                        pacer_error_analysis[f'pacer_type_{pacer_type}'] = {
                            'count': len(pacer_errors),
                            'mean_error': round(pacer_errors.mean(), 4),
                            'std_error': round(pacer_errors.std(), 4),
                            'median_error': round(pacer_errors.median(), 4)
                        }
                results['pacer_error_correlation'] = pacer_error_analysis

            logger.info(f"Pacer analysis completed: {len(pacer_columns)} pacer columns analyzed")
            return results

        except Exception as e:
            logger.error(f"Error in pacer analysis: {e}")
            return {"error": f"Pacer analysis failed: {str(e)}"}

def run_comprehensive_analysis(vector_db_path: str = "vector_db",
                             sqlite_db_path: str = "rmsai_metadata.db",
                             output_dir: str = "analytics_output",
                             selected_leads: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run comprehensive analytics pipeline"""
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting comprehensive analytics pipeline...")
    if selected_leads:
        logger.info(f"Analytics restricted to leads: {selected_leads}")

    analytics = EmbeddingAnalytics(vector_db_path, sqlite_db_path, selected_leads)

    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_results': {}
    }

    try:
        # 1. Clustering analysis
        logger.info("Running clustering analysis...")
        clusters = analytics.discover_embedding_clusters()
        results['analysis_results']['clustering'] = clusters

        # 2. Anomaly detection
        logger.info("Running anomaly detection...")
        anomalies = analytics.detect_anomalous_patterns()
        results['analysis_results']['anomaly_detection'] = anomalies

        # 3. Temporal analysis
        logger.info("Running temporal analysis...")
        temporal = analytics.temporal_pattern_analysis()
        results['analysis_results']['temporal_patterns'] = temporal

        # 4. Similarity network
        logger.info("Generating similarity network...")
        network = analytics.generate_similarity_network()
        results['analysis_results']['similarity_network'] = network

        # 5. Pacer pattern analysis
        logger.info("Running pacer pattern analysis...")
        pacer_analysis = analytics.analyze_pacer_patterns()
        results['analysis_results']['pacer_patterns'] = pacer_analysis

        # 6. Visualization
        logger.info("Creating visualizations...")
        viz_path = os.path.join(output_dir, "embedding_visualization.png")
        viz_results = analytics.visualize_embedding_space(save_path=viz_path)
        results['analysis_results']['visualization'] = viz_results

        # Save results to JSON
        results_path = os.path.join(output_dir, "analytics_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Comprehensive analysis complete. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        results['error'] = str(e)

    return results

if __name__ == "__main__":
    # Example usage
    results = run_comprehensive_analysis()

    if 'error' not in results:
        print("Analytics Results Summary:")
        print("=" * 50)

        if 'clustering' in results['analysis_results']:
            clustering = results['analysis_results']['clustering']
            if 'kmeans' in clustering:
                print(f"K-means: {clustering['kmeans']['n_clusters']} clusters")
            if 'dbscan' in clustering:
                print(f"DBSCAN: {clustering['dbscan']['n_clusters']} clusters")

        if 'anomaly_detection' in results['analysis_results']:
            anomaly = results['analysis_results']['anomaly_detection']
            if 'consensus' in anomaly:
                print(f"Consensus anomalies: {anomaly['consensus']['n_anomalies']}")

        if 'similarity_network' in results['analysis_results']:
            network = results['analysis_results']['similarity_network']
            if 'n_similarity_groups' in network:
                print(f"Similarity groups: {network['n_similarity_groups']}")

        print("\nFull results saved to analytics_output/analytics_results.json")
    else:
        print(f"Analysis failed: {results['error']}")