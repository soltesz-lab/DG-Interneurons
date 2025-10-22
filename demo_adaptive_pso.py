"""
Demonstration of Adaptive PSO functionality.

This script shows various usage patterns for the AdaptivePSO class.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from adaptive_pso import AdaptivePSO, PSOConfig, BenchmarkFunctions


def demo_basic_usage():
    """Demonstrate basic PSO usage on sphere function."""
    print("="*80)
    print("DEMO 1: Basic Usage - Sphere Function")
    print("="*80)
    
    # Define problem
    bounds = [(-5.0, 5.0)] * 10
    
    # Create PSO with default configuration
    config = PSOConfig(
        n_particles=32,
        max_iterations=50,
        verbose=True
    )
    
    pso = AdaptivePSO(
        objective_function=BenchmarkFunctions.sphere,
        bounds=bounds,
        config=config,
        random_seed=42
    )
    
    # Optimize
    result = pso.optimize()
    
    print(f"\nFinal Results:")
    print(f"  Best score: {result.best_score:.6f}")
    print(f"  Total evaluations: {result.n_evaluations}")
    print(f"  Convergence iteration: {result.convergence_iteration}")
    
    return result


def demo_feature_comparison():
    """Compare different PSO configurations."""
    print("\n" + "="*80)
    print("DEMO 2: Feature Comparison")
    print("="*80)
    
    bounds = [(-5.0, 5.0)] * 10
    
    configurations = {
        'Standard PSO': PSOConfig(
            n_particles=32, max_iterations=30,
            use_obl_initialization=False,
            use_adaptive_parameters=False,
            use_multi_swarm=False,
            verbose=False
        ),
        'OBL Only': PSOConfig(
            n_particles=32, max_iterations=30,
            use_obl_initialization=True,
            use_adaptive_parameters=False,
            use_multi_swarm=False,
            verbose=False
        ),
        'OBL + Adaptive': PSOConfig(
            n_particles=32, max_iterations=30,
            use_obl_initialization=True,
            use_adaptive_parameters=True,
            use_multi_swarm=False,
            verbose=False
        ),
        'Full Features': PSOConfig(
            n_particles=32, max_iterations=30,
            use_obl_initialization=True,
            use_adaptive_parameters=True,
            use_multi_swarm=True,
            n_sub_swarms=8,
            verbose=False
        )
    }
    
    results = {}
    
    for name, config in configurations.items():
        print(f"\nRunning: {name}")
        pso = AdaptivePSO(
            BenchmarkFunctions.rastrigin,  # Challenging multimodal function
            bounds, config, random_seed=42
        )
        result = pso.optimize()
        results[name] = result
        print(f"  Best score: {result.best_score:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    for name, result in results.items():
        plt.semilogy(result.history['best_scores'], label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Score (log scale)')
    plt.title('PSO Configuration Comparison on Rastrigin Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pso_outputs/pso_comparison.png', dpi=150)
    print(f"\nComparison plot saved to pso_outputs/pso_comparison.png")
    
    return results


def demo_benchmark_suite():
    """Test on multiple benchmark functions."""
    print("\n" + "="*80)
    print("DEMO 3: Benchmark Function Suite")
    print("="*80)
    
    bounds = [(-5.0, 5.0)] * 10
    config = PSOConfig(
        n_particles=32, max_iterations=50,
        verbose=False
    )
    
    functions = {
        'Sphere': BenchmarkFunctions.sphere,
        'Rosenbrock': BenchmarkFunctions.rosenbrock,
        'Rastrigin': BenchmarkFunctions.rastrigin,
        'Ackley': BenchmarkFunctions.ackley,
        'Griewank': BenchmarkFunctions.griewank,
    }
    
    results = {}
    
    for name, func in functions.items():
        print(f"\nOptimizing {name}...")
        pso = AdaptivePSO(func, bounds, config, random_seed=42)
        result = pso.optimize()
        results[name] = result
        print(f"  Best score: {result.best_score:.6f}")
        print(f"  Evaluations: {result.n_evaluations}")
    
    # Plot convergence curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        ax.semilogy(result.history['best_scores'], linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Score')
        ax.set_title(f'{name} Function')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('pso_outputs/benchmark_suite.png', dpi=150)
    print(f"\nBenchmark suite plot saved to pso_outputs/benchmark_suite.png")
    
    return results


def demo_diversity_tracking():
    """Demonstrate diversity tracking and adaptive behavior."""
    print("\n" + "="*80)
    print("DEMO 4: Diversity Tracking")
    print("="*80)
    
    bounds = [(-5.0, 5.0)] * 10
    config = PSOConfig(
        n_particles=32, max_iterations=50,
        use_adaptive_parameters=True,
        verbose=False
    )
    
    pso = AdaptivePSO(BenchmarkFunctions.rastrigin, bounds, config, random_seed=42)
    result = pso.optimize()
    
    # Extract parameter history
    iterations = range(len(result.history['diversity']))
    diversity = result.history['diversity']
    w_values = [p['w'] for p in result.history['parameters']]
    c1_values = [p['c1'] for p in result.history['parameters']]
    c2_values = [p['c2'] for p in result.history['parameters']]
    
    # Plot diversity and parameters
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Diversity
    axes[0].plot(iterations, diversity, 'b-', linewidth=2)
    axes[0].set_ylabel('Swarm Diversity')
    axes[0].set_title('Diversity Evolution')
    axes[0].grid(True, alpha=0.3)
    
    # Inertia weight
    axes[1].plot(iterations, w_values, 'r-', linewidth=2)
    axes[1].set_ylabel('Inertia Weight (w)')
    axes[1].set_title('Adaptive Inertia Weight')
    axes[1].grid(True, alpha=0.3)
    
    # Cognitive and social parameters
    axes[2].plot(iterations, c1_values, 'g-', linewidth=2, label='c1 (cognitive)')
    axes[2].plot(iterations, c2_values, 'm-', linewidth=2, label='c2 (social)')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Parameter Value')
    axes[2].set_title('Adaptive Cognitive/Social Parameters')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pso_outputs/diversity_tracking.png', dpi=150)
    print(f"\nDiversity tracking plot saved to pso_outputs/diversity_tracking.png")
    print(f"\nFinal Results:")
    print(f"  Best score: {result.best_score:.6f}")
    print(f"  Initial diversity: {diversity[0]:.6f}")
    print(f"  Final diversity: {diversity[-1]:.6f}")
    
    return result


def demo_robustness_test():
    """Test robustness across multiple runs with different seeds."""
    print("\n" + "="*80)
    print("DEMO 5: Robustness Test (Multiple Seeds)")
    print("="*80)
    
    bounds = [(-5.0, 5.0)] * 10
    config = PSOConfig(
        n_particles=32, max_iterations=30,
        verbose=False
    )
    
    n_runs = 10
    results = []
    
    print(f"\nRunning {n_runs} optimizations with different seeds...")
    for seed in range(n_runs):
        pso = AdaptivePSO(BenchmarkFunctions.rastrigin, bounds, config, 
                         random_seed=seed)
        result = pso.optimize()
        results.append(result.best_score)
        print(f"  Seed {seed}: {result.best_score:.6f}")
    
    # Statistics
    mean_score = np.mean(results)
    std_score = np.std(results)
    min_score = np.min(results)
    max_score = np.max(results)
    
    print(f"\nStatistics:")
    print(f"  Mean: {mean_score:.6f}")
    print(f"  Std:  {std_score:.6f}")
    print(f"  Min:  {min_score:.6f}")
    print(f"  Max:  {max_score:.6f}")
    print(f"  CV:   {std_score/mean_score:.3f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=15, edgecolor='black', alpha=0.7)
    plt.axvline(mean_score, color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_score:.2f}')
    plt.xlabel('Best Score')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Best Scores over {n_runs} Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pso_outputs/robustness_test.png', dpi=150)
    print(f"\nRobustness test plot saved to pso_outputs/robustness_test.png")
    
    return results


def demo_custom_objective():
    """Demonstrate using PSO with a custom objective function."""
    print("\n" + "="*80)
    print("DEMO 6: Custom Objective Function")
    print("="*80)
    
    # Define a custom objective: minimize sum of absolute values + penalty
    def custom_objective(x):
        # x is shape (n_particles, n_dimensions)
        sum_abs = np.sum(np.abs(x), axis=1)
        # Add penalty for being far from origin
        distance = np.sqrt(np.sum(x**2, axis=1))
        return sum_abs + 0.5 * distance
    
    bounds = [(-3.0, 3.0)] * 5
    config = PSOConfig(
        n_particles=16, max_iterations=30,
        verbose=False
    )
    
    print("\nOptimizing custom objective...")
    pso = AdaptivePSO(custom_objective, bounds, config, random_seed=42)
    result = pso.optimize()
    
    print(f"\nResults:")
    print(f"  Best score: {result.best_score:.6f}")
    print(f"  Best position: {result.best_position}")
    print(f"  Expected optimum: [0, 0, 0, 0, 0]")
    
    return result


if __name__ == "__main__":
    print("\nAdaptive PSO Demonstration Suite")
    print("="*80)

    output_dir = "pso_outputs"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Run all demonstrations
    demo_basic_usage()
    demo_feature_comparison()
    demo_benchmark_suite()
    demo_diversity_tracking()
    demo_robustness_test()
    demo_custom_objective()
    
    print("\n" + "="*80)
    print("All demonstrations complete!")
    print("="*80)
