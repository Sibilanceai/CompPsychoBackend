import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from typing import Dict, List, Optional
import logging
from collections import defaultdict
import seaborn as sns
import json
import os
import re
from scipy.stats import entropy as scipy_entropy
import csv
from pathlib import Path
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_entropy(probs):
    """Calculate entropy for a probability distribution."""
    # Ensure probabilities are normalized and non-negative
    probs = np.array(probs).flatten()
    probs = np.abs(probs)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    return scipy_entropy(probs + 1e-10)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def load_character_aliases(file_path):
    """Load character aliases from CSV file."""
    character_aliases = []
    try:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                aliases = [clean_character_name(name) for name in row 
                          for name in re.split(r'[,;]\s*', name) if name]
                if aliases:
                    character_aliases.append(aliases)
    except Exception as e:
        logger.error(f"Error loading character aliases: {e}")
        return []
    return character_aliases

def clean_character_name(name):
    """Clean character names consistently."""
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip().lower()

def load_saved_data():
    """Load data saved by TE_graph_app.py."""
    logger.info("Loading saved data...")
    
    # Load character information
    file_path = '../data_collection/characters_list.csv'
    characters = load_character_aliases(file_path)
    primary_names = [aliases[0] for aliases in characters if aliases]
    
    # Load graphs
    with open('graph_data.json', 'r') as f:
        graph_data = json.load(f)
        
    # Convert graph data back to networkx graphs
    all_graphs = []
    for time_step in graph_data:
        G = nx.DiGraph()
        elements = time_step['elements']
        
        # Add nodes
        nodes = [e for e in elements if 'source' not in e['data']]
        for node in nodes:
            G.add_node(node['data']['id'], 
                      label=node['data']['label'],
                      group=node['data']['group'])
        
        # Add edges
        edges = [e for e in elements if 'source' in e['data']]
        for edge in edges:
            G.add_edge(edge['data']['source'], 
                      edge['data']['target'], 
                      weight=edge['data']['weight'])
            
        all_graphs.append(G)
    
    # Load matrices and timestamps for each character
    time_series_matrices = {}
    time_stamps = {}
    
    for char in primary_names:
        try:
            matrices = np.load(f'transition_matrices_{char}.npy', allow_pickle=True)
            stamps = np.load(f'time_stamps_{char}.npy', allow_pickle=True)
            time_series_matrices[char] = matrices
            time_stamps[char] = stamps
        except Exception as e:
            logger.error(f"Error loading data for {char}: {e}")
            continue
    
    # Get all time points
    all_time_points = sorted(set().union(*[set(stamps) for stamps in time_stamps.values()]))
    
    return primary_names, time_series_matrices, time_stamps, all_time_points, all_graphs

class BigFiveAnalyzer:
    def __init__(self, characters: List[str], time_series_matrices: Dict, 
                 time_stamps: Dict, all_time_points: List[float], 
                 narrative_graphs: List[nx.Graph]):
        """Initialize analyzer with loaded data."""
        self.characters = characters
        self.time_series_matrices = time_series_matrices
        self.time_stamps = time_stamps
        self.all_time_points = all_time_points
        self.narrative_graphs = narrative_graphs
        
        self.categories = ['high-level', 'task-specific', 'context-specific']
        self.terms = ['short-term', 'medium-term', 'long-term']
        
        self.personality_scores = defaultdict(lambda: defaultdict(dict))
        self.temporal_traits = defaultdict(lambda: defaultdict(list))
    
    def analyze_personality(self, window_size: int = 50):
        """Run personality analysis on loaded data."""
        logger.info("Starting personality analysis...")
        
        for char in self.characters:
            logger.info(f"Analyzing character: {char}")
            
            char_matrices = self.time_series_matrices[char]
            char_stamps = self.time_stamps[char]
            
            for t in range(0, len(self.all_time_points) - window_size + 1):
                window_start = self.all_time_points[t]
                window_end = self.all_time_points[t + window_size - 1]
                
                window_matrices = self._get_window_data(char_matrices, char_stamps, 
                                                      window_start, window_end)
                window_graphs = self.narrative_graphs[t:t+window_size]
                
                traits = self._calculate_window_traits(char, window_matrices, window_graphs)
                
                for trait, score in traits.items():
                    self.temporal_traits[char][trait].append({
                        'time': window_start,
                        'score': score
                    })
            
            self._calculate_final_scores(char)
    
    def _calculate_window_traits(self, char: str, window_matrices: List[Dict], 
                               window_graphs: List[nx.Graph]) -> Dict[str, float]:
        """Calculate Big Five traits for a specific time window."""
        # Initialize trait scores
        trait_scores = {}
        
        # Calculate Openness (cognitive flexibility)
        trait_scores['openness'] = self._calculate_openness(window_matrices)
        
        # Calculate Conscientiousness (behavioral consistency)
        trait_scores['conscientiousness'] = self._calculate_conscientiousness(window_matrices)
        
        # Calculate Extraversion (social network centrality)
        trait_scores['extraversion'] = self._calculate_extraversion(char, window_graphs)
        
        # Calculate Agreeableness (cooperative behavior)
        trait_scores['agreeableness'] = self._calculate_agreeableness(char, window_graphs)
        
        # Calculate Neuroticism (behavioral volatility)
        trait_scores['neuroticism'] = self._calculate_neuroticism(window_matrices)
        
        return trait_scores
    
    def _normalize_score(self, score: float, min_val: float = 0.0, 
                    max_val: float = 1.0, center: float = None) -> float:
        """
        Normalize score using sigmoid function with optional centering.
        
        Args:
            score: Raw score to normalize
            min_val: Minimum value for normalized score
            max_val: Maximum value for normalized score
            center: Optional center point for sigmoid (defaults to midpoint)
        """
        if np.isnan(score) or np.isinf(score):
            return (min_val + max_val) / 2
            
        if center is None:
            center = (min_val + max_val) / 2
            
        # Scale factor for sigmoid
        scale = 5.0  # Adjust for steeper/gentler sigmoid
        
        # Center and normalize score
        x = scale * (score - center)
        sigmoid = 1 / (1 + np.exp(-x))
        
        # Scale to desired range
        return min_val + sigmoid * (max_val - min_val)

    def _calculate_openness(self, matrices: List[Dict]) -> float:
        """Calculate openness from cognitive flexibility in transitions."""
        if not matrices:
            return 0.5
            
        flexibility_scores = []
        
        for i in range(len(matrices)-1):
            curr_mat = matrices[i]
            next_mat = matrices[i+1]
            
            for category in self.categories:
                for term in self.terms:
                    if category in curr_mat and category in next_mat:
                        if term in curr_mat[category] and term in next_mat[category]:
                            current = curr_mat[category][term]
                            future = next_mat[category][term]
                            
                            # Calculate exploration score
                            nonzero_curr = np.count_nonzero(current)
                            nonzero_next = np.count_nonzero(future)
                            exploration = (nonzero_next - nonzero_curr) / current.size
                            
                            # Calculate transition diversity
                            unique_transitions = len(np.unique(current)) + len(np.unique(future))
                            diversity = unique_transitions / (2 * current.size)
                            
                            # Calculate entropy change
                            curr_entropy = calculate_entropy(current)
                            next_entropy = calculate_entropy(future)
                            entropy_change = abs(next_entropy - curr_entropy)
                            
                            # Calculate behavioral flexibility
                            curr_nonzero_indices = set(np.nonzero(current.flatten())[0])
                            next_nonzero_indices = set(np.nonzero(future.flatten())[0])
                            new_transitions = len(next_nonzero_indices - curr_nonzero_indices)
                            flexibility = new_transitions / current.size
                            
                            # Combine metrics
                            score = (0.3 * exploration + 
                                   0.2 * diversity + 
                                   0.2 * entropy_change +
                                   0.3 * flexibility)
                            
                            flexibility_scores.append(score)
    
        if not flexibility_scores:
            return 0.5
            
        avg_flexibility = np.mean(flexibility_scores)
        logger.debug(f"Average flexibility score: {avg_flexibility}")
        
        # Normalize to a reasonable range for openness
        return self._normalize_score(avg_flexibility, min_val=0.2, max_val=0.8)

    def _calculate_conscientiousness(self, matrices: List[Dict]) -> float:
        """Calculate conscientiousness from behavioral consistency."""
        if not matrices:
            return 0.5
            
        consistency_scores = []
        
        for i in range(len(matrices)-1):
            curr_mat = matrices[i]
            next_mat = matrices[i+1]
            
            for category in self.categories:
                for term in self.terms:
                    if category in curr_mat and category in next_mat:
                        if term in curr_mat[category] and term in next_mat[category]:
                            current = curr_mat[category][term]
                            future = next_mat[category][term]
                            
                            # Calculate behavioral consistency
                            change = np.linalg.norm(future - current)
                            max_change = np.sqrt(2)  # Maximum possible change for normalized matrices
                            consistency = 1 - (change / max_change)
                            
                            # Calculate pattern stability
                            curr_pattern = np.argmax(current, axis=1)
                            next_pattern = np.argmax(future, axis=1)
                            pattern_stability = np.mean(curr_pattern == next_pattern)
                            
                            # Calculate decision certainty
                            curr_certainty = np.mean(np.max(current, axis=1))
                            next_certainty = np.mean(np.max(future, axis=1))
                            certainty = np.mean([curr_certainty, next_certainty])
                            
                            # Combine metrics
                            score = (0.4 * consistency + 
                                   0.3 * pattern_stability + 
                                   0.3 * certainty)
                            
                            consistency_scores.append(score)
        
        if not consistency_scores:
            return 0.5
            
        avg_consistency = np.mean(consistency_scores)
        logger.debug(f"Average consistency score: {avg_consistency}")
        
        return self._normalize_score(avg_consistency)

    def _calculate_extraversion(self, char: str, graphs: List[nx.Graph]) -> float:
        """Calculate extraversion from network centrality metrics."""
        if not graphs:
            logger.warning(f"No graphs available for {char}")
            return 0.5
            
        centrality_scores = []
        
        for graph in graphs:
            try:
                # Get all nodes for this character (including different categories/terms)
                char_nodes = [node for node in graph.nodes() if char in node]
                
                if not char_nodes:
                    continue
                    
                # Calculate metrics for each node associated with the character
                node_scores = []
                for node in char_nodes:
                    # Calculate normalized centrality measures
                    degree = len(list(graph.neighbors(node))) / (len(graph.nodes()) - 1)
                    
                    # Calculate betweenness more efficiently
                    paths = nx.single_source_shortest_path_length(graph, node)
                    betweenness = len(paths) / (len(graph.nodes()) * (len(graph.nodes()) - 1))
                    
                    # Use degree centrality if eigenvector fails
                    try:
                        eigenvector = nx.eigenvector_centrality_numpy(graph)[node]
                    except:
                        eigenvector = degree
                    
                    # Combine measures with weights
                    score = 0.4 * degree + 0.3 * betweenness + 0.3 * eigenvector
                    node_scores.append(score)
                
                if node_scores:
                    centrality_scores.append(np.mean(node_scores))
                    
            except Exception as e:
                logger.debug(f"Error calculating extraversion for {char} in graph: {e}")
                continue
        
        if not centrality_scores:
            logger.warning(f"No valid centrality scores calculated for {char}")
            return 0.5
            
        avg_centrality = np.mean(centrality_scores)
        logger.debug(f"Average centrality for {char}: {avg_centrality}")
        return self._normalize_score(avg_centrality)

    def _calculate_agreeableness(self, char: str, graphs: List[nx.Graph]) -> float:
        """Calculate agreeableness from cooperative network behavior."""
        if not graphs:
            logger.warning(f"No graphs available for {char}")
            return 0.5
            
        cooperation_scores = []
        
        for graph in graphs:
            try:
                # Get all nodes for this character
                char_nodes = [node for node in graph.nodes() if char in node]
                
                if not char_nodes:
                    continue
                    
                # Calculate metrics for each node
                for node in char_nodes:
                    # Calculate clustering coefficient
                    neighbors = list(graph.neighbors(node))
                    if len(neighbors) > 1:
                        clustering = nx.clustering(graph, node)
                        
                        # Calculate reciprocity
                        reciprocal = sum(1 for n in neighbors if graph.has_edge(n, node))
                        reciprocal_ratio = reciprocal / len(neighbors)
                        
                        # Calculate interaction stability
                        edge_weights = [graph[node][n].get('weight', 0) for n in neighbors]
                        weight_stability = 1 - np.std(edge_weights) if edge_weights else 0
                        
                        # Combine metrics
                        score = (0.4 * clustering + 
                                0.4 * reciprocal_ratio + 
                                0.2 * weight_stability)
                        cooperation_scores.append(score)
                        
            except Exception as e:
                logger.debug(f"Error calculating agreeableness for {char} in graph: {e}")
                continue
        
        if not cooperation_scores:
            logger.warning(f"No valid cooperation scores calculated for {char}")
            return 0.5
            
        avg_cooperation = np.mean(cooperation_scores)
        logger.debug(f"Average cooperation for {char}: {avg_cooperation}")
        return self._normalize_score(avg_cooperation)

    def _calculate_neuroticism(self, matrices: List[Dict]) -> float:
        """Calculate neuroticism from behavioral volatility."""
        if not matrices:
            return 0.0
            
        volatility_scores = []
        
        for i in range(len(matrices)-1):
            curr_mat = matrices[i]
            next_mat = matrices[i+1]
            
            for category in self.categories:
                for term in self.terms:
                    if category in curr_mat and category in next_mat:
                        if term in curr_mat[category] and term in next_mat[category]:
                            current = curr_mat[category][term]
                            future = next_mat[category][term]
                            
                            # Calculate normalized volatility
                            change_rate = np.linalg.norm(future - current)
                            max_change = np.sqrt(2)  # Maximum possible change for normalized matrices
                            
                            volatility = change_rate / max_change
                            volatility_scores.append(volatility)
        
        avg_volatility = np.mean(volatility_scores) if volatility_scores else 0.0
        return self._normalize_score(avg_volatility)
    
    def _get_window_data(self, matrices, stamps, start_time, end_time):
        """Get matrices for a specific time window."""
        window_indices = [i for i, t in enumerate(stamps) 
                         if start_time <= t <= end_time]
        return [matrices[i] for i in window_indices]
    
    def _calculate_final_scores(self, char: str):
        """Calculate final trait scores from temporal patterns."""
        for trait in ['openness', 'conscientiousness', 'extraversion', 
                     'agreeableness', 'neuroticism']:
            scores = [entry['score'] for entry in self.temporal_traits[char][trait]]
            
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                self.personality_scores[char][trait] = {
                    'score': mean_score,
                    'stability': 1 / (1 + std_score),
                    'temporal_pattern': scores
                }
            else:
                self.personality_scores[char][trait] = {
                    'score': 0.0,
                    'stability': 0.0,
                    'temporal_pattern': []
                }
    
    def _convert_to_serializable(self, data):
        """Convert numpy types to Python native types."""
        if isinstance(data, dict):
            return {key: self._convert_to_serializable(value) 
                   for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data

    def save_results(self, output_dir: str = 'results'):
        """Save analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert results to serializable format
        results = {
            char: {
                'traits': self._convert_to_serializable(dict(self.personality_scores[char])),
                'temporal_evolution': {
                    trait: [self._convert_to_serializable(entry) for entry in values]
                    for trait, values in self.temporal_traits[char].items()
                }
            }
            for char in self.characters
        }
        
        # Save with custom encoder
        with open(f'{output_dir}/personality_analysis.json', 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        
        # Also save as CSV for easier reading
        traits_df = pd.DataFrame([
            {
                'character': char,
                'trait': trait,
                'score': self.personality_scores[char][trait]['score'],
                'stability': self.personality_scores[char][trait]['stability']
            }
            for char in self.characters
            for trait in ['openness', 'conscientiousness', 'extraversion', 
                         'agreeableness', 'neuroticism']
        ])
        
        traits_df.to_csv(f'{output_dir}/personality_traits.csv', index=False)
        
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        print("\nPersonality Analysis Summary:")
        for char in self.characters:
            print(f"\nCharacter: {char}")
            for trait in ['openness', 'conscientiousness', 'extraversion', 
                         'agreeableness', 'neuroticism']:
                score = float(self.personality_scores[char][trait]['score'])
                stability = float(self.personality_scores[char][trait]['stability'])
                print(f"{trait.capitalize()}: {score:.3f} (stability: {stability:.3f})")

def main():
    """Main execution function."""
    # Load saved data
    primary_names, time_series_matrices, time_stamps, all_time_points, all_graphs = load_saved_data()
    
    # Initialize analyzer
    analyzer = BigFiveAnalyzer(
        characters=primary_names,
        time_series_matrices=time_series_matrices,
        time_stamps=time_stamps,
        all_time_points=all_time_points,
        narrative_graphs=all_graphs
    )
    
    # Run analysis
    analyzer.analyze_personality()
    
    # Save results
    analyzer.save_results()

if __name__ == "__main__":
    main()
