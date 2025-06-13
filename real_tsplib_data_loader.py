"""
🚀 Revolutionary Real TSPLIB Data Loader
=======================================

This replaces the FAKE data with real TSPLIB problems and optimal solutions!

Features:
- Uses real .tsp files with tsplib95
- Real optimal tour lengths (berlin52: 7542, kroA100: 21282, etc.)
- Graph construction from node coordinates  
- Proper distance matrices using EUC_2D/EXPLICIT weights
- Scales from small (16 nodes) to massive (33K+ nodes)
- Ready for breakthrough training!
"""

import torch
import tsplib95
import json
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import random


class RevolutionaryTSPLIBLoader:
    """🎯 Real TSPLIB data loader - No more fake targets!"""
    
    def __init__(self, 
                 tsplib_dir: str = "testing/data/TSPLIB/tsplib_downloads",
                 optimal_solutions_path: str = "testing/data/TSPLIB/tsplib_optimal_solutions.json",
                 difficulty_filter: Optional[str] = None):
        """
        Args:
            tsplib_dir: Directory containing .tsp files
            optimal_solutions_path: JSON with real optimal tour lengths
            difficulty_filter: "easy" (≤100 nodes), "medium" (≤500), "hard" (≤2000), "extreme" (>2000), None (all)
        """
        self.tsplib_dir = Path(tsplib_dir)
        self.optimal_solutions_path = Path(optimal_solutions_path)
        self.difficulty_filter = difficulty_filter
        
        # Load optimal solutions
        with open(self.optimal_solutions_path, 'r') as f:
            self.optimal_solutions = json.load(f)
        
        # Discover available problems
        self.problems = self._discover_problems()
        
        print(f"🔥 Revolutionary TSPLIB Loader initialized!")
        print(f"   📁 Directory: {self.tsplib_dir}")
        print(f"   🎯 Problems found: {len(self.problems)}")
        print(f"   🏆 Optimal solutions: {len(self.optimal_solutions)}")
        if difficulty_filter:
            print(f"   🎚️  Difficulty filter: {difficulty_filter}")
        
        # Show examples
        self._show_examples()
    
    def _discover_problems(self) -> List[Dict]:
        """Discover all available TSPLIB problems"""
        problems = []
        
        for tsp_file in self.tsplib_dir.glob("*.tsp"):
            problem_name = tsp_file.stem
            
            # Skip if no optimal solution available
            if problem_name not in self.optimal_solutions:
                continue
            
            try:
                # Load problem to get basic info
                problem = tsplib95.load(str(tsp_file))
                dimension = problem.dimension
                
                # Apply difficulty filter
                if self.difficulty_filter:
                    if self.difficulty_filter == "easy" and dimension > 100:
                        continue
                    elif self.difficulty_filter == "medium" and (dimension <= 100 or dimension > 500):
                        continue
                    elif self.difficulty_filter == "hard" and (dimension <= 500 or dimension > 2000):
                        continue
                    elif self.difficulty_filter == "extreme" and dimension <= 2000:
                        continue
                
                problems.append({
                    'name': problem_name,
                    'file_path': str(tsp_file),
                    'dimension': dimension,
                    'optimal_tour_length': self.optimal_solutions[problem_name],
                    'comment': getattr(problem, 'comment', ''),
                    'edge_weight_type': getattr(problem, 'edge_weight_type', 'UNKNOWN')
                })
                
            except Exception as e:
                print(f"⚠️  Skipping {problem_name}: {e}")
                continue
        
        # Sort by difficulty (dimension)
        problems.sort(key=lambda x: x['dimension'])
        return problems
    
    def _show_examples(self):
        """Show example problems loaded"""
        print("\n🎯 Example problems loaded:")
        print("=" * 70)
        
        # Show range of difficulties
        for i, example_type in enumerate(["easiest", "medium", "hardest"]):
            if example_type == "easiest" and len(self.problems) > 0:
                prob = self.problems[0]
            elif example_type == "medium" and len(self.problems) > 1:
                prob = self.problems[len(self.problems)//2]
            elif example_type == "hardest" and len(self.problems) > 0:
                prob = self.problems[-1]
            else:
                continue
                
            print(f"{example_type.upper():>8}: {prob['name']:>12} | "
                  f"{prob['dimension']:>4} nodes | "
                  f"optimal: {prob['optimal_tour_length']:>8} | "
                  f"{prob['edge_weight_type']}")
        
        print("=" * 70)
    
    def get_problem_info(self) -> List[Dict]:
        """Get info about all available problems"""
        return self.problems.copy()
    
    def load_problem(self, problem_name: str) -> Tuple[Data, float, Dict]:
        """
        Load a specific TSPLIB problem
        
        Returns:
            graph: PyTorch Geometric Data object with node features and edges
            optimal_tour_length: Real optimal tour length
            metadata: Problem information
        """
        # Find problem
        problem_info = None
        for prob in self.problems:
            if prob['name'] == problem_name:
                problem_info = prob
                break
        
        if problem_info is None:
            raise ValueError(f"Problem '{problem_name}' not found or not available")
        
        # Load with tsplib95
        problem = tsplib95.load(problem_info['file_path'])
        
        # Extract node coordinates and create graph
        graph_data = self._create_graph_from_problem(problem)
        
        return graph_data, problem_info['optimal_tour_length'], problem_info
    
    def _create_graph_from_problem(self, problem) -> Data:
        """Create PyTorch Geometric graph from TSPLIB problem"""
        
        # Get nodes
        nodes = list(problem.get_nodes())
        num_nodes = len(nodes)
        
        # Extract node features
        if hasattr(problem, 'node_coords') and problem.node_coords:
            # Use coordinates as features
            coords = []
            for node in nodes:
                if node in problem.node_coords:
                    coords.append(problem.node_coords[node])
                else:
                    coords.append([0.0, 0.0])  # Fallback
            
            node_features = torch.tensor(coords, dtype=torch.float32)
            
            # Normalize coordinates to [0, 1] range for better training
            if node_features.numel() > 0:
                min_vals = node_features.min(dim=0)[0]
                max_vals = node_features.max(dim=0)[0]
                range_vals = max_vals - min_vals
                range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
                node_features = (node_features - min_vals) / range_vals
        else:
            # Fallback: random features
            node_features = torch.randn(num_nodes, 2)
        
        # Create complete graph (TSP connects all cities)
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Add edge weights based on distances
        edge_weights = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    try:
                        # Use TSPLIB's get_weight method for accurate distances
                        weight = problem.get_weight(nodes[i], nodes[j])
                        edge_weights.append(weight)
                    except:
                        # Fallback: Euclidean distance
                        if node_features.size(1) >= 2:
                            dist = torch.norm(node_features[i] - node_features[j]).item()
                            edge_weights.append(dist)
                        else:
                            edge_weights.append(1.0)
        
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_batch(self, batch_size: int = 8, shuffle: bool = True) -> List[Tuple[Data, float, Dict]]:
        """Get a batch of problems"""
        if shuffle:
            selected_problems = random.sample(self.problems, 
                                            min(batch_size, len(self.problems)))
        else:
            selected_problems = self.problems[:batch_size]
        
        batch = []
        for prob_info in selected_problems:
            try:
                graph, optimal, metadata = self.load_problem(prob_info['name'])
                batch.append((graph, optimal, metadata))
            except Exception as e:
                print(f"⚠️  Failed to load {prob_info['name']}: {e}")
                continue
        
        return batch
    
    def create_curriculum_batches(self, curriculum_stages: int = 4) -> List[List[Dict]]:
        """Create curriculum learning batches (easy to hard)"""
        # Sort problems by difficulty
        sorted_problems = sorted(self.problems, key=lambda x: x['dimension'])
        
        # Split into curriculum stages
        problems_per_stage = len(sorted_problems) // curriculum_stages
        stages = []
        
        for i in range(curriculum_stages):
            start_idx = i * problems_per_stage
            if i == curriculum_stages - 1:
                # Last stage gets remaining problems
                stage_problems = sorted_problems[start_idx:]
            else:
                end_idx = (i + 1) * problems_per_stage
                stage_problems = sorted_problems[start_idx:end_idx]
            
            stages.append(stage_problems)
            
            # Show stage info
            if stage_problems:
                min_nodes = min(p['dimension'] for p in stage_problems)
                max_nodes = max(p['dimension'] for p in stage_problems)
                print(f"📚 Curriculum Stage {i+1}: {len(stage_problems)} problems, "
                      f"{min_nodes}-{max_nodes} nodes")
        
        return stages


def test_revolutionary_loader():
    """Test the revolutionary TSPLIB loader"""
    print("🧪 Testing Revolutionary TSPLIB Loader...")
    
    # Test with easy problems first
    loader = RevolutionaryTSPLIBLoader(difficulty_filter="easy")
    
    if len(loader.problems) == 0:
        print("❌ No problems found! Check your paths.")
        return
    
    # Test loading a specific problem
    problem_name = loader.problems[0]['name']
    print(f"\n🔬 Testing problem: {problem_name}")
    
    graph, optimal_length, metadata = loader.load_problem(problem_name)
    
    print(f"✅ Graph created:")
    print(f"   📊 Nodes: {graph.x.size(0)}")
    print(f"   🔗 Edges: {graph.edge_index.size(1)}")
    print(f"   📏 Node features: {graph.x.size(1)}")
    print(f"   🎯 Optimal tour length: {optimal_length}")
    print(f"   📝 Comment: {metadata['comment']}")
    
    # Test batch loading
    print(f"\n🔬 Testing batch loading...")
    batch = loader.get_batch(batch_size=3)
    print(f"✅ Batch loaded with {len(batch)} problems")
    
    for i, (g, opt, meta) in enumerate(batch):
        print(f"   Problem {i+1}: {meta['name']} ({g.x.size(0)} nodes, optimal: {opt})")


if __name__ == "__main__":
    test_revolutionary_loader() 