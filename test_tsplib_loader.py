"""
🧪 Test TSPLIB Data Loading (No Torch Required)
==============================================

Simple test to verify we can load real TSPLIB data and optimal solutions.
"""

import tsplib95
import json
from pathlib import Path


def test_tsplib_loading():
    """Test loading TSPLIB data with your existing files"""
    print("🧪 Testing TSPLIB Data Loading...")
    print("=" * 50)
    
    # Paths to your data
    tsplib_dir = Path("testing/data/TSPLIB/tsplib_downloads")
    optimal_solutions_path = Path("testing/data/TSPLIB/tsplib_optimal_solutions.json")
    
    # Load optimal solutions
    print(f"📁 Loading optimal solutions from: {optimal_solutions_path}")
    with open(optimal_solutions_path, 'r') as f:
        optimal_solutions = json.load(f)
    
    print(f"✅ Loaded {len(optimal_solutions)} optimal solutions")
    
    # Show some examples
    print("\n🎯 Example optimal solutions:")
    examples = ["berlin52", "kroA100", "eil51", "gr17", "burma14"]
    for name in examples:
        if name in optimal_solutions:
            print(f"   {name:>10}: {optimal_solutions[name]:>8}")
    
    # Find available .tsp files
    print(f"\n📁 Scanning for .tsp files in: {tsplib_dir}")
    tsp_files = list(tsplib_dir.glob("*.tsp"))
    print(f"✅ Found {len(tsp_files)} .tsp files")
    
    # Test loading a few problems
    print(f"\n🔬 Testing problem loading...")
    test_problems = ["berlin52", "kroA100", "eil51", "gr17"]
    
    for problem_name in test_problems:
        tsp_file = tsplib_dir / f"{problem_name}.tsp"
        
        if not tsp_file.exists():
            print(f"⚠️  {problem_name}.tsp not found")
            continue
            
        if problem_name not in optimal_solutions:
            print(f"⚠️  {problem_name} optimal solution not found")
            continue
        
        try:
            # Load with tsplib95
            problem = tsplib95.load(str(tsp_file))
            
            print(f"✅ {problem_name}:")
            print(f"   📊 Dimension: {problem.dimension}")
            print(f"   🎯 Optimal: {optimal_solutions[problem_name]}")
            print(f"   📝 Comment: {getattr(problem, 'comment', 'N/A')}")
            print(f"   🔗 Edge Weight Type: {getattr(problem, 'edge_weight_type', 'N/A')}")
            
            # Test coordinate extraction
            if hasattr(problem, 'node_coords') and problem.node_coords:
                coords_count = len(problem.node_coords)
                print(f"   📍 Coordinates: {coords_count} nodes")
                
                # Show first few coordinates
                nodes = list(problem.get_nodes())[:3]
                for node in nodes:
                    if node in problem.node_coords:
                        coord = problem.node_coords[node]
                        print(f"      Node {node}: ({coord[0]}, {coord[1]})")
            
            # Test distance calculation
            nodes = list(problem.get_nodes())
            if len(nodes) >= 2:
                try:
                    dist = problem.get_weight(nodes[0], nodes[1])
                    print(f"   📏 Distance {nodes[0]}-{nodes[1]}: {dist}")
                except Exception as e:
                    print(f"   ⚠️  Distance calculation failed: {e}")
            
            print()
            
        except Exception as e:
            print(f"❌ Failed to load {problem_name}: {e}")
    
    print("🎉 TSPLIB loading test completed!")
    print("=" * 50)
    
    # Summary
    available_problems = []
    for tsp_file in tsp_files:
        problem_name = tsp_file.stem
        if problem_name in optimal_solutions:
            available_problems.append(problem_name)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total .tsp files: {len(tsp_files)}")
    print(f"   With optimal solutions: {len(available_problems)}")
    print(f"   Ready for training: {len(available_problems)}")
    
    # Show difficulty distribution
    easy = medium = hard = extreme = 0
    for problem_name in available_problems:
        tsp_file = tsplib_dir / f"{problem_name}.tsp"
        try:
            problem = tsplib95.load(str(tsp_file))
            dim = problem.dimension
            if dim <= 100:
                easy += 1
            elif dim <= 500:
                medium += 1
            elif dim <= 2000:
                hard += 1
            else:
                extreme += 1
        except:
            continue
    
    print(f"\n🎚️  DIFFICULTY DISTRIBUTION:")
    print(f"   Easy (≤100 nodes): {easy}")
    print(f"   Medium (≤500 nodes): {medium}")
    print(f"   Hard (≤2000 nodes): {hard}")
    print(f"   Extreme (>2000 nodes): {extreme}")


if __name__ == "__main__":
    test_tsplib_loading() 