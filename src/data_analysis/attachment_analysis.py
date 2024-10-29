import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy.interpolate import interp1d
from collections import defaultdict
import json
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from community import community_louvain
from node2vec import Node2Vec

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttachmentAnalyzer:
    def __init__(self, characters: List[str], transition_matrices: Dict[str, List[Dict]], 
                 time_stamps: Dict[str, List[int]], graph_snapshots: Optional[List[nx.Graph]] = None):
        """
        Initialize AttachmentAnalyzer with character and matrix information.
        
        Args:
            characters: List of character names
            transition_matrices: Dictionary of transition matrices by character
            time_stamps: Dictionary of timestamps by character
            graph_snapshots: Optional list of network snapshots over time
        """
        self.characters = characters
        self.transition_matrices = transition_matrices
        self.time_stamps = time_stamps
        self.graph_snapshots = graph_snapshots or []
        self.interaction_network = nx.DiGraph()
        self.proximity_scores = {}
        self.safe_haven_scores = {}
        self.secure_base_scores = {}
        self.attachment_styles = {}
        self.temporal_patterns = defaultdict(list)
        self.embeddings = {}
        
    def analyze_network_evolution(self):
        """Analyze the evolution of network structure over time."""
        if not self.graph_snapshots:
            logger.warning("No graph snapshots available for evolution analysis")
            return
        
        evolution_metrics = {
            'density': [],
            'clustering': [],
            'path_length': [],
            'modularity': []
        }
        
        for snapshot in self.graph_snapshots:
            evolution_metrics['density'].append(nx.density(snapshot))
            evolution_metrics['clustering'].append(nx.average_clustering(snapshot))
            
            # Path length for connected components only
            if nx.is_connected(snapshot):
                path_length = nx.average_shortest_path_length(snapshot)
            else:
                path_length = float('inf')
            evolution_metrics['path_length'].append(path_length)
            
            # Community detection
            partition = community_louvain.best_partition(snapshot)
            modularity = community_louvain.modularity(partition, snapshot)
            evolution_metrics['modularity'].append(modularity)
            
        return evolution_metrics
    
    def compute_embeddings(self):
        """Compute node embeddings for each character over time."""
        if not self.graph_snapshots:
            return
        
        for t, snapshot in enumerate(self.graph_snapshots):
            node2vec = Node2Vec(snapshot, dimensions=64, walk_length=30, num_walks=200, workers=4)
            model = node2vec.fit(window=10, min_count=1)
            
            # Store embeddings for each character
            for char in self.characters:
                if char not in self.embeddings:
                    self.embeddings[char] = []
                try:
                    self.embeddings[char].append(model.wv[char])
                except KeyError:
                    # Character not in this snapshot
                    self.embeddings[char].append(np.zeros(64))
    
    def analyze_attachment_dynamics(self, window_size: int = 10):
        """
        Analyze temporal dynamics of attachment patterns.
        
        Args:
            window_size: Size of sliding window for analysis
        """
        for t in range(len(self.graph_snapshots) - window_size + 1):
            window = self.graph_snapshots[t:t+window_size]
            
            # Calculate attachment metrics for window
            proximity = self._calculate_proximity_window(window)
            safe_haven = self._calculate_safe_haven_window(window)
            secure_base = self._calculate_secure_base_window(window)
            
            # Store temporal patterns
            self.temporal_patterns['proximity'].append(proximity)
            self.temporal_patterns['safe_haven'].append(safe_haven)
            self.temporal_patterns['secure_base'].append(secure_base)
    
    def _calculate_proximity_window(self, window: List[nx.Graph]) -> Dict[str, float]:
        """Calculate proximity scores within a time window."""
        proximity_scores = defaultdict(list)
        
        for graph in window:
            for char in self.characters:
                if char in graph:
                    # Calculate degree centrality as proxy for proximity seeking
                    score = nx.degree_centrality(graph)[char]
                    proximity_scores[char].append(score)
                    
        # Average scores over window
        return {char: np.mean(scores) for char, scores in proximity_scores.items()}
    
    def _calculate_safe_haven_window(self, window: List[nx.Graph]) -> Dict[str, float]:
        """Calculate safe haven scores within a time window."""
        safe_haven_scores = defaultdict(list)
        
        for graph in window:
            for char in self.characters:
                if char in graph:
                    # Calculate weighted clustering coefficient as proxy for safe haven behavior
                    score = nx.clustering(graph, char)
                    safe_haven_scores[char].append(score)
                    
        return {char: np.mean(scores) for char, scores in safe_haven_scores.items()}
    
    def _calculate_secure_base_window(self, window: List[nx.Graph]) -> Dict[str, float]:
        """Calculate secure base scores within a time window."""
        secure_base_scores = defaultdict(list)
        
        for graph in window:
            for char in self.characters:
                if char in graph:
                    # Calculate betweenness centrality as proxy for secure base behavior
                    score = nx.betweenness_centrality(graph)[char]
                    secure_base_scores[char].append(score)
                    
        return {char: np.mean(scores) for char, scores in secure_base_scores.items()}
    
    def analyze_attachment_stability(self):
        """Analyze the stability of attachment patterns over time."""
        stability_metrics = {}
        
        for char in self.characters:
            # Calculate variance in attachment metrics over time
            proximity_stability = np.var(self.temporal_patterns['proximity'])
            safe_haven_stability = np.var(self.temporal_patterns['safe_haven'])
            secure_base_stability = np.var(self.temporal_patterns['secure_base'])
            
            stability_metrics[char] = {
                'proximity_stability': proximity_stability,
                'safe_haven_stability': safe_haven_stability,
                'secure_base_stability': secure_base_stability,
                'overall_stability': np.mean([
                    proximity_stability,
                    safe_haven_stability,
                    secure_base_stability
                ])
            }
            
        return stability_metrics
    
    def classify_attachment_patterns(self):
        """Classify attachment patterns using clustering on temporal features."""
        # Extract temporal features for each character
        features = []
        for char in self.characters:
            char_features = []
            # Add temporal stability features
            stability_metrics = self.analyze_attachment_stability()[char]
            char_features.extend([
                stability_metrics['proximity_stability'],
                stability_metrics['safe_haven_stability'],
                stability_metrics['secure_base_stability']
            ])
            # Add embedding features if available
            if char in self.embeddings:
                char_features.extend(np.mean(self.embeddings[char], axis=0))
            features.append(char_features)
            
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        # Map clusters to attachment styles
        styles = ['Secure', 'Anxious', 'Avoidant', 'Disorganized']
        self.attachment_styles = {char: styles[label] for char, label in zip(self.characters, labels)}
        
        return self.attachment_styles
    
    def visualize_attachment_evolution(self, save_path: Optional[str] = None):
        """Visualize the evolution of attachment patterns."""
        # Plot network evolution metrics
        evolution_metrics = self.analyze_network_evolution()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        for (metric, values), ax in zip(evolution_metrics.items(), axes.flat):
            ax.plot(values, marker='o')
            ax.set_title(f'{metric.capitalize()} Evolution')
            ax.set_xlabel('Time')
            ax.set_ylabel(metric.capitalize())
            
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/network_evolution.png")
        plt.close()
        
        # Plot temporal patterns
        fig, ax = plt.subplots(figsize=(12, 6))
        for metric, values in self.temporal_patterns.items():
            ax.plot(values, label=metric.capitalize())
        ax.set_title('Attachment Pattern Evolution')
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Score')
        ax.legend()
        
        if save_path:
            plt.savefig(f"{save_path}/attachment_patterns.png")
        plt.close()
    
    def export_comprehensive_results(self, filepath: str):
        """Export comprehensive analysis results to JSON."""
        results = {
            'attachment_styles': self.attachment_styles,
            'temporal_patterns': {k: v.tolist() for k, v in self.temporal_patterns.items()},
            'stability_metrics': self.analyze_attachment_stability(),
            'network_evolution': self.analyze_network_evolution(),
            'final_network_metrics': {
                'density': nx.density(self.interaction_network),
                'clustering': nx.average_clustering(self.interaction_network),
                'modularity': community_louvain.modularity(
                    community_louvain.best_partition(self.interaction_network),
                    self.interaction_network
                )
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your data
    analyzer = AttachmentAnalyzer(
        characters=primary_names,
        transition_matrices=time_series_matrices,
        time_stamps=time_stamps,
        graph_snapshots=all_graphs  # From your existing graph generation
    )
    
    # Perform comprehensive analysis
    analyzer.analyze_network_evolution()
    analyzer.compute_embeddings()
    analyzer.analyze_attachment_dynamics()
    attachment_styles = analyzer.classify_attachment_patterns()
    
    # Visualize results
    analyzer.visualize_attachment_evolution(save_path='./results')
    
    # Export comprehensive results
    analyzer.export_comprehensive_results('./results/comprehensive_analysis.json')