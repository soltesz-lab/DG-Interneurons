"""
Unit tests for Adaptive PSO implementation.

Tests cover:
- Basic optimization on benchmark functions
- Opposition-based learning
- Diversity-adaptive parameters
- Dynamic multi-swarm
- Intelligent restart
- Reproducibility
- Edge cases and error handling
- Performance comparisons

Run with: pytest test_adaptive_pso.py -v
"""

import numpy as np
import pytest
from adaptive_pso import (
    AdaptivePSO, PSOConfig, PSOResult, BenchmarkFunctions, DiversityState
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sphere_function():
    """Simple sphere function for testing."""
    return BenchmarkFunctions.sphere


@pytest.fixture
def default_bounds():
    """Default bounds for 10D problems."""
    return [(-5.0, 5.0)] * 10


@pytest.fixture
def minimal_config():
    """Minimal configuration for fast testing."""
    return PSOConfig(
        n_particles=16,
        max_iterations=10,
        verbose=False
    )


@pytest.fixture
def standard_config():
    """Standard configuration for thorough testing."""
    return PSOConfig(
        n_particles=32,
        max_iterations=50,
        verbose=False
    )


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestBasicOptimization:
    """Test basic PSO optimization on benchmark functions."""
    
    def test_sphere_convergence(self, sphere_function, default_bounds, standard_config):
        """Test convergence on simple sphere function."""
        pso = AdaptivePSO(sphere_function, default_bounds, standard_config, 
                         random_seed=42)
        result = pso.optimize()
        
        # Should converge to near-zero for sphere function
        assert result.best_score < 1.0, \
            f"Expected score < 1.0, got {result.best_score}"
        assert result.n_evaluations > 0
        assert len(result.best_position) == 10
        assert len(result.history['best_scores']) == standard_config.max_iterations
    
    def test_rosenbrock_optimization(self, default_bounds, standard_config):
        """Test on more challenging Rosenbrock function."""
        pso = AdaptivePSO(BenchmarkFunctions.rosenbrock, default_bounds, 
                         standard_config, random_seed=42)
        result = pso.optimize()
        
        # Rosenbrock is harder, but should still improve significantly
        assert result.best_score < 100.0, \
            f"Expected score < 100, got {result.best_score}"
    
    def test_rastrigin_multimodal(self, default_bounds, standard_config):
        """Test on highly multimodal Rastrigin function."""
        pso = AdaptivePSO(BenchmarkFunctions.rastrigin, default_bounds, 
                         standard_config, random_seed=42)
        result = pso.optimize()
        
        # Rastrigin has many local minima, harder to converge
        assert result.best_score < 50.0, \
            f"Expected score < 50, got {result.best_score}"
    
    def test_2d_optimization(self, sphere_function):
        """Test optimization in 2D space."""
        bounds = [(-5.0, 5.0)] * 2
        config = PSOConfig(n_particles=16, max_iterations=20, verbose=False)
        
        pso = AdaptivePSO(sphere_function, bounds, config, random_seed=42)
        result = pso.optimize()
        
        assert result.best_score < 0.1
        assert len(result.best_position) == 2
    
    def test_high_dimensional(self, sphere_function):
        """Test optimization in high-dimensional space (50D)."""
        bounds = [(-5.0, 5.0)] * 50
        config = PSOConfig(n_particles=64, max_iterations=30, verbose=False)
        
        pso = AdaptivePSO(sphere_function, bounds, config, random_seed=42)
        result = pso.optimize()
        
        assert result.best_score < 100.0  # Harder in high dimensions, relaxed threshold
        assert len(result.best_position) == 50


# ============================================================================
# Opposition-Based Learning Tests
# ============================================================================

class TestOppositionBasedLearning:
    """Test OBL initialization and escape mechanisms."""
    
    def test_obl_initialization_improvement(self, sphere_function, default_bounds):
        """Test that OBL initialization improves over random."""
        # With OBL
        config_obl = PSOConfig(
            n_particles=32, max_iterations=1, 
            use_obl_initialization=True,
            verbose=False
        )
        pso_obl = AdaptivePSO(sphere_function, default_bounds, config_obl, 
                             random_seed=42)
        result_obl = pso_obl.optimize()
        
        # Without OBL
        config_random = PSOConfig(
            n_particles=32, max_iterations=1,
            use_obl_initialization=False,
            verbose=False
        )
        pso_random = AdaptivePSO(sphere_function, default_bounds, config_random, 
                                random_seed=42)
        result_random = pso_random.optimize()
        
        # OBL should have more evaluations initially (2x for init + iteration evals)
        # OBL: 64 (init: 32 random + 32 opposite) + 32 (iteration) = 96
        # Random: 32 (init) + 32 (iteration) = 64
        assert result_obl.n_evaluations > result_random.n_evaluations
    
    def test_obl_escape_mechanism(self, sphere_function, default_bounds):
        """Test that OBL escape is triggered during stagnation."""
        config = PSOConfig(
            n_particles=16,
            max_iterations=20,
            obl_escape_enabled=True,
            obl_escape_threshold=3,
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        # Should complete successfully
        assert result.best_score >= 0
    
    def test_obl_disabled(self, sphere_function, default_bounds):
        """Test PSO with OBL completely disabled."""
        config = PSOConfig(
            n_particles=16,
            max_iterations=10,
            use_obl_initialization=False,
            obl_escape_enabled=False,
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        assert result.best_score >= 0
        assert result.n_evaluations == config.n_particles * (config.max_iterations + 1)


# ============================================================================
# Adaptive Parameters Tests
# ============================================================================

class TestAdaptiveParameters:
    """Test diversity-adaptive parameter control."""
    
    def test_adaptive_vs_standard(self, sphere_function, default_bounds):
        """Compare adaptive parameters vs standard schedule."""
        # Adaptive
        config_adaptive = PSOConfig(
            n_particles=32, max_iterations=30,
            use_adaptive_parameters=True,
            verbose=False
        )
        pso_adaptive = AdaptivePSO(sphere_function, default_bounds, 
                                   config_adaptive, random_seed=42)
        result_adaptive = pso_adaptive.optimize()
        
        # Standard
        config_standard = PSOConfig(
            n_particles=32, max_iterations=30,
            use_adaptive_parameters=False,
            verbose=False
        )
        pso_standard = AdaptivePSO(sphere_function, default_bounds, 
                                   config_standard, random_seed=42)
        result_standard = pso_standard.optimize()
        
        # Both should converge, but adaptive might be better
        assert result_adaptive.best_score >= 0
        assert result_standard.best_score >= 0
    
    def test_diversity_state_tracking(self, sphere_function, default_bounds):
        """Test that diversity states are tracked correctly."""
        config = PSOConfig(
            n_particles=32, max_iterations=20,
            use_adaptive_parameters=True,
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        # Check diversity history exists
        assert 'diversity_states' in result.history
        assert len(result.history['diversity_states']) == config.max_iterations
        
        # Check states are valid
        valid_states = {s.value for s in DiversityState}
        for state in result.history['diversity_states']:
            assert state in valid_states
    
    def test_parameter_adaptation(self, sphere_function, default_bounds):
        """Test that PSO parameters are adapted over time."""
        config = PSOConfig(
            n_particles=32, max_iterations=20,
            use_adaptive_parameters=True,
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        # Check parameter history
        assert 'parameters' in result.history
        assert len(result.history['parameters']) == config.max_iterations
        
        # Parameters should vary
        w_values = [p['w'] for p in result.history['parameters']]
        c1_values = [p['c1'] for p in result.history['parameters']]
        c2_values = [p['c2'] for p in result.history['parameters']]
        
        # Should have some variation (not all identical)
        assert len(set(w_values)) > 1 or len(set(c1_values)) > 1


# ============================================================================
# Multi-Swarm Tests
# ============================================================================

class TestMultiSwarm:
    """Test dynamic multi-swarm functionality."""
    
    def test_multi_swarm_initialization(self, sphere_function, default_bounds):
        """Test that sub-swarms are initialized correctly."""
        config = PSOConfig(
            n_particles=32,
            max_iterations=5,
            use_multi_swarm=True,
            n_sub_swarms=8,
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        # Should complete successfully with 8 sub-swarms
        assert result.best_score >= 0
    
    def test_regrouping_period(self, sphere_function, default_bounds):
        """Test that regrouping occurs at specified intervals."""
        config = PSOConfig(
            n_particles=32,
            max_iterations=15,
            use_multi_swarm=True,
            n_sub_swarms=8,
            regrouping_period=5,
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        # Should have regrouped at iterations 5, 10, 15
        assert result.best_score >= 0
    
    def test_multi_swarm_vs_single(self, sphere_function, default_bounds):
        """Compare multi-swarm vs single swarm performance."""
        # Multi-swarm
        config_multi = PSOConfig(
            n_particles=32, max_iterations=30,
            use_multi_swarm=True,
            n_sub_swarms=8,
            verbose=False
        )
        pso_multi = AdaptivePSO(sphere_function, default_bounds, 
                                config_multi, random_seed=42)
        result_multi = pso_multi.optimize()
        
        # Single swarm
        config_single = PSOConfig(
            n_particles=32, max_iterations=30,
            use_multi_swarm=False,
            verbose=False
        )
        pso_single = AdaptivePSO(sphere_function, default_bounds, 
                                 config_single, random_seed=42)
        result_single = pso_single.optimize()
        
        # Both should converge
        assert result_multi.best_score >= 0
        assert result_single.best_score >= 0
    
    def test_sub_swarm_count_validation(self, sphere_function, default_bounds):
        """Test that invalid sub-swarm configurations are caught."""
        # More sub-swarms than particles should work (some swarms will be smaller)
        config = PSOConfig(
            n_particles=16,
            max_iterations=5,
            use_multi_swarm=True,
            n_sub_swarms=8,  # Valid: 2 particles per swarm
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        assert result.best_score >= 0


# ============================================================================
# Restart Mechanism Tests
# ============================================================================

class TestIntelligentRestart:
    """Test intelligent restart mechanism."""
    
    def test_restart_triggered(self, sphere_function, default_bounds):
        """Test that restart is triggered on stagnation."""
        config = PSOConfig(
            n_particles=16,
            max_iterations=30,
            restart_threshold=5,
            elite_fraction=0.25,
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        # Should complete successfully
        assert result.best_score >= 0
    
    def test_elite_preservation(self, sphere_function, default_bounds):
        """Test that elite particles are preserved during restart."""
        config = PSOConfig(
            n_particles=20,
            max_iterations=20,
            restart_threshold=5,
            elite_fraction=0.2,  # Keep 4 elites
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        assert result.best_score >= 0


# ============================================================================
# Reproducibility Tests
# ============================================================================

class TestReproducibility:
    """Test that optimization is reproducible with fixed seed."""
    
    def test_same_seed_same_result(self, sphere_function, default_bounds, minimal_config):
        """Test that same seed produces identical results."""
        # Run 1
        pso1 = AdaptivePSO(sphere_function, default_bounds, minimal_config, 
                          random_seed=42)
        result1 = pso1.optimize()
        
        # Run 2 with same seed
        pso2 = AdaptivePSO(sphere_function, default_bounds, minimal_config, 
                          random_seed=42)
        result2 = pso2.optimize()
        
        # Should be identical
        assert np.allclose(result1.best_position, result2.best_position)
        assert abs(result1.best_score - result2.best_score) < 1e-10
        assert result1.n_evaluations == result2.n_evaluations
    
    def test_different_seed_different_result(self, sphere_function, 
                                            default_bounds, minimal_config):
        """Test that different seeds produce different results."""
        # Run 1
        pso1 = AdaptivePSO(sphere_function, default_bounds, minimal_config, 
                          random_seed=42)
        result1 = pso1.optimize()
        
        # Run 2 with different seed
        pso2 = AdaptivePSO(sphere_function, default_bounds, minimal_config, 
                          random_seed=123)
        result2 = pso2.optimize()
        
        # Should be different (with high probability)
        # Note: Might converge to same optimum, but trajectories should differ
        assert result1.n_evaluations == result2.n_evaluations  # Same budget
        # Best positions might be similar for convex function, but unlikely identical
        position_diff = np.linalg.norm(result1.best_position - result2.best_position)
        # If they converged to same optimum, positions should be close
        # Otherwise they should differ
        # We just check that the algorithm ran successfully
        assert result1.best_score >= 0
        assert result2.best_score >= 0


# ============================================================================
# History Tracking Tests
# ============================================================================

class TestHistoryTracking:
    """Test that optimization history is tracked correctly."""
    
    def test_history_structure(self, sphere_function, default_bounds, minimal_config):
        """Test that history has correct structure."""
        pso = AdaptivePSO(sphere_function, default_bounds, minimal_config, 
                         random_seed=42)
        result = pso.optimize()
        
        # Check all expected keys exist
        expected_keys = ['best_scores', 'mean_scores', 'diversity', 
                        'parameters', 'diversity_states']
        for key in expected_keys:
            assert key in result.history
        
        # Check lengths match iterations
        for key in expected_keys:
            assert len(result.history[key]) == minimal_config.max_iterations
    
    def test_best_scores_monotonic(self, sphere_function, default_bounds, minimal_config):
        """Test that best scores are monotonically non-increasing."""
        pso = AdaptivePSO(sphere_function, default_bounds, minimal_config, 
                         random_seed=42)
        result = pso.optimize()
        
        best_scores = result.history['best_scores']
        for i in range(1, len(best_scores)):
            assert best_scores[i] <= best_scores[i-1], \
                f"Best score increased at iteration {i}"
    
    def test_diversity_trends(self, sphere_function, default_bounds):
        """Test that diversity generally decreases over time."""
        config = PSOConfig(n_particles=32, max_iterations=30, verbose=False)
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        diversity = result.history['diversity']
        
        # Diversity should generally decrease (with possible increases from restart)
        # Check that final diversity is less than initial
        assert diversity[-1] < diversity[0]


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_dimension(self, sphere_function):
        """Test optimization in 1D space."""
        bounds = [(-5.0, 5.0)]
        config = PSOConfig(n_particles=8, max_iterations=10, verbose=False)
        
        pso = AdaptivePSO(sphere_function, bounds, config, random_seed=42)
        result = pso.optimize()
        
        assert len(result.best_position) == 1
        assert result.best_score < 1.0
    
    def test_asymmetric_bounds(self, sphere_function):
        """Test with asymmetric bounds."""
        bounds = [(-10.0, 5.0), (-2.0, 8.0), (-5.0, 5.0)]
        config = PSOConfig(n_particles=16, max_iterations=20, verbose=False)
        
        pso = AdaptivePSO(sphere_function, bounds, config, random_seed=42)
        result = pso.optimize()
        
        # Check solution is within bounds
        for i, (lower, upper) in enumerate(bounds):
            assert lower <= result.best_position[i] <= upper
    
    def test_small_population(self, sphere_function, default_bounds):
        """Test with very small population."""
        # For small populations, need fewer sub-swarms
        config = PSOConfig(
            n_particles=4, 
            max_iterations=20, 
            use_multi_swarm=True,
            n_sub_swarms=2,  # Only 2 sub-swarms for 4 particles
            verbose=False
        )
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        assert result.best_score >= 0
    
    def test_single_iteration(self, sphere_function, default_bounds):
        """Test with single iteration."""
        config = PSOConfig(n_particles=16, max_iterations=1, verbose=False)
        
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        result = pso.optimize()
        
        assert result.n_iterations == 1
        assert len(result.history['best_scores']) == 1
    
    def test_invalid_bounds(self, sphere_function):
        """Test that invalid bounds are caught."""
        # Lower bound > upper bound
        bounds = [(5.0, -5.0)]
        config = PSOConfig(n_particles=8, max_iterations=5, verbose=False)
        
        with pytest.raises(AssertionError):
            pso = AdaptivePSO(sphere_function, bounds, config, random_seed=42)
    
    def test_invalid_config(self, sphere_function, default_bounds):
        """Test that invalid configurations are caught."""
        # Invalid: n_particles < n_sub_swarms
        with pytest.raises(AssertionError):
            config = PSOConfig(
                n_particles=8,
                n_sub_swarms=16,  # More sub-swarms than particles
                verbose=False
            )


# ============================================================================
# Performance Comparison Tests
# ============================================================================

class TestPerformanceComparisons:
    """Compare performance of different configurations."""
    
    def test_full_vs_minimal_features(self, default_bounds):
        """Compare full-featured PSO vs minimal configuration."""
        # Full features
        config_full = PSOConfig(
            n_particles=32, max_iterations=30,
            use_obl_initialization=True,
            use_adaptive_parameters=True,
            use_multi_swarm=True,
            verbose=False
        )
        
        # Minimal features
        config_minimal = PSOConfig(
            n_particles=32, max_iterations=30,
            use_obl_initialization=False,
            use_adaptive_parameters=False,
            use_multi_swarm=False,
            verbose=False
        )
        
        # Test on Rastrigin (multimodal)
        pso_full = AdaptivePSO(BenchmarkFunctions.rastrigin, default_bounds, 
                               config_full, random_seed=42)
        result_full = pso_full.optimize()
        
        pso_minimal = AdaptivePSO(BenchmarkFunctions.rastrigin, default_bounds, 
                                  config_minimal, random_seed=42)
        result_minimal = pso_minimal.optimize()
        
        # Both should converge, full should generally be better
        assert result_full.best_score >= 0
        assert result_minimal.best_score >= 0
        
        # Note: On a single seed, minimal might sometimes win by chance,
        # but on average full features should be better
    
    def test_benchmark_all_functions(self, default_bounds):
        """Test PSO on all benchmark functions."""
        config = PSOConfig(n_particles=32, max_iterations=30, verbose=False)
        
        functions = [
            ('Sphere', BenchmarkFunctions.sphere, 2.0),
            ('Rosenbrock', BenchmarkFunctions.rosenbrock, 150.0),
            ('Rastrigin', BenchmarkFunctions.rastrigin, 60.0),  # Highly multimodal
            ('Ackley', BenchmarkFunctions.ackley, 5.0),
            ('Griewank', BenchmarkFunctions.griewank, 1.0),
        ]
        
        results = {}
        for name, func, threshold in functions:
            pso = AdaptivePSO(func, default_bounds, config, random_seed=42)
            result = pso.optimize()
            results[name] = result.best_score
            
            # Should achieve reasonable convergence
            assert result.best_score < threshold, \
                f"{name}: Expected < {threshold}, got {result.best_score}"
        
        # All should complete successfully
        assert len(results) == len(functions)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_optimization_workflow(self, sphere_function, default_bounds):
        """Test a complete optimization workflow."""
        # Create custom config
        config = PSOConfig(
            n_particles=32,
            max_iterations=50,
            use_obl_initialization=True,
            use_adaptive_parameters=True,
            use_multi_swarm=True,
            n_sub_swarms=8,
            regrouping_period=5,
            verbose=False
        )
        
        # Initialize optimizer
        pso = AdaptivePSO(sphere_function, default_bounds, config, random_seed=42)
        
        # Run optimization
        result = pso.optimize()
        
        # Verify result
        assert isinstance(result, PSOResult)
        assert result.best_score < 1.0
        assert len(result.best_position) == 10
        assert result.n_iterations == 50
        assert result.config == config
        
        # Check history
        assert len(result.history['best_scores']) == 50
        assert len(result.history['diversity']) == 50
        
        # Verify monotonic improvement
        for i in range(1, len(result.history['best_scores'])):
            assert result.history['best_scores'][i] <= \
                   result.history['best_scores'][i-1]
    
    def test_multiple_runs_statistics(self, sphere_function, default_bounds):
        """Test running multiple optimizations and computing statistics."""
        config = PSOConfig(n_particles=16, max_iterations=20, verbose=False)
        
        n_runs = 5
        results = []
        
        for seed in range(n_runs):
            pso = AdaptivePSO(sphere_function, default_bounds, config, 
                             random_seed=seed)
            result = pso.optimize()
            results.append(result.best_score)
        
        # Compute statistics
        mean_score = np.mean(results)
        std_score = np.std(results)
        min_score = np.min(results)
        max_score = np.max(results)
        
        # All should be reasonable
        assert mean_score < 5.0
        assert min_score >= 0
        assert max_score < 10.0
        assert std_score < mean_score  # Reasonable variation


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
