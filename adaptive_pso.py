"""
Adaptive Particle Swarm Optimization.

This module provides a PSO implementation with:
- Opposition-Based Learning (OBL) for initialization and escape
- Diversity-adaptive parameter control
- Dynamic Multi-Swarm (DMS-PSO) with regrouping
- Intelligent restart mechanisms
- Optional metadata tracking from objective function

The objective function can return either:
- np.ndarray: scores only
- Tuple[np.ndarray, List[Dict]]: (scores, metadata) for each evaluation
"""

import logging
import pprint
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class DiversityState(Enum):
    """Swarm diversity states for adaptive parameter control."""
    LOW = "low"       # < low_threshold: needs exploration
    MEDIUM = "medium" # between thresholds: balanced
    HIGH = "high"     # > high_threshold: needs exploitation


@dataclass
class PSOConfig:
    """
    Configuration for Adaptive PSO algorithm.
    
    Attributes:
        n_particles: Population size
        max_iterations: Maximum number of iterations
        w_max, w_min: Inertia weight bounds
        c1_range, c2_range: Cognitive and social parameter ranges
        use_obl_initialization: Enable opposition-based initialization
        obl_escape_enabled: Enable OBL escape from stagnation
        obl_escape_threshold: Iterations without improvement before OBL escape
        obl_escape_fraction: Fraction of worst particles to replace
        use_adaptive_parameters: Enable diversity-adaptive parameters
        diversity_low_threshold: Threshold for low diversity state
        diversity_high_threshold: Threshold for high diversity state
        use_multi_swarm: Enable dynamic multi-swarm
        n_sub_swarms: Number of sub-swarms for DMS-PSO
        regrouping_period: Iterations between regrouping
        restart_threshold: Iterations without improvement before restart
        elite_fraction: Fraction of best particles to preserve in restart
        restart_std_factor: Std of Gaussian noise for restart (relative to bounds)
        diagnostic_frequency: Print diagnostics every N new bests
        verbose: Enable detailed output
        track_metadata: Whether to track and store metadata from objective function
    """
    
    # Basic PSO parameters
    n_particles: int = 32
    max_iterations: int = 50
    w_max: float = 0.9
    w_min: float = 0.2
    c1_range: Tuple[float, float] = (1.5, 2.5)
    c2_range: Tuple[float, float] = (1.5, 2.5)
    
    # Opposition-Based Learning
    use_obl_initialization: bool = True
    obl_escape_enabled: bool = True
    obl_escape_threshold: int = 5
    obl_escape_fraction: float = 0.25
    
    # Diversity-Adaptive Parameters
    use_adaptive_parameters: bool = True
    diversity_low_threshold: float = 0.2
    diversity_high_threshold: float = 0.6
    
    # Dynamic Multi-Swarm
    use_multi_swarm: bool = True
    n_sub_swarms: int = 8
    regrouping_period: int = 5
    
    # Intelligent Restart
    restart_threshold: int = 8
    elite_fraction: float = 0.25
    restart_std_factor: float = 0.2
    
    # Diagnostics
    diagnostic_frequency: int = 5
    verbose: bool = True
    
    # Metadata tracking
    track_metadata: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.n_particles > 0, "n_particles must be positive"
        assert self.max_iterations > 0, "max_iterations must be positive"
        assert 0 < self.w_min < self.w_max < 1, "Invalid inertia weights"
        assert self.n_sub_swarms > 0, "n_sub_swarms must be positive"
        assert self.n_particles >= self.n_sub_swarms, \
            "n_particles must be >= n_sub_swarms"
        assert 0 < self.elite_fraction < 1, "elite_fraction must be in (0, 1)"
        assert 0 < self.obl_escape_fraction < 1, "obl_escape_fraction must be in (0, 1)"


@dataclass
class PSOResult:
    """
    Results from PSO optimization.
    
    Attributes:
        best_position: Best position found
        best_score: Best score achieved
        best_metadata: Metadata for best configuration (if available)
        history: Optimization history with scores, diversity, parameters
        metadata_history: Metadata for all new best solutions (if tracking enabled)
        n_iterations: Number of iterations performed
        n_evaluations: Total function evaluations
        n_new_bests: Number of times global best was updated
        final_diversity: Final swarm diversity
        convergence_iteration: Iteration where convergence occurred (if any)
        config: Configuration used for optimization
    """
    
    best_position: np.ndarray
    best_score: float
    best_metadata: Optional[Dict[str, Any]] = None
    history: Dict[str, List] = field(default_factory=lambda: {
        'best_scores': [],
        'mean_scores': [],
        'diversity': [],
        'parameters': []
    })
    metadata_history: List[Dict[str, Any]] = field(default_factory=list)
    n_iterations: int = 0
    n_evaluations: int = 0
    n_new_bests: int = 0
    final_diversity: float = 0.0
    convergence_iteration: Optional[int] = None
    config: Optional[PSOConfig] = None


class AdaptivePSO:
    """
    Adaptive Particle Swarm Optimization with metadata tracking
    
    The objective function can return either:
    1. np.ndarray of scores
    2. Tuple[np.ndarray, List[Dict]] with (scores, metadata for each evaluation)
    
    Lower scores are better (minimization).
    
    Features:
        - Opposition-Based Learning (OBL) for better initialization and escape
        - Diversity-adaptive parameter control (w, c1, c2)
        - Dynamic Multi-Swarm (DMS-PSO) with periodic regrouping
        - Intelligent restart mechanism preserving elites
        - Comprehensive tracking and diagnostics
        - Optional metadata tracking for troubleshooting**
    
    Example with metadata:
        >>> def objective(positions: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        ...     scores = np.sum(positions**2, axis=1)
        ...     metadata = [{'position_norm': np.linalg.norm(p)} for p in positions]
        ...     return scores, metadata
        >>> 
        >>> bounds = [(-5.0, 5.0)] * 10
        >>> config = PSOConfig(track_metadata=True)
        >>> pso = AdaptivePSO(objective, bounds, config)
        >>> result = pso.optimize()
        >>> print(f"Best score: {result.best_score:.6f}")
        >>> print(f"Best metadata: {result.best_metadata}")
    """
    
    def __init__(self, 
                 objective_function: Callable[[np.ndarray], Union[np.ndarray, Tuple[np.ndarray, List[Dict]]]],
                 bounds: List[Tuple[float, float]],
                 config: Optional[PSOConfig] = None,
                 random_seed: Optional[int] = None,
                 print_metadata: Optional[Callable[[np.ndarray, float, Optional[Dict]], None]] = None):
        """
        Initialize Adaptive PSO optimizer.
        
        Args:
            objective_function: Function that evaluates a batch of positions.
                                Can return either:
                                - np.ndarray of shape (n_particles,) with scores
                                - Tuple[np.ndarray, List[Dict]] with (scores, metadata)
                                Lower scores are better.
            bounds: List of (lower, upper) tuples for each dimension
            config: PSO configuration (uses defaults if None)
            random_seed: Random seed for reproducibility
            print_metadata: Optional custom metadata printer, to be invoked during diagnostics
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_dimensions = len(bounds)
        self.config = config if config is not None else PSOConfig()
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Extract bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])
        
        # Validate bounds
        assert np.all(self.lower_bounds < self.upper_bounds), \
            "Lower bounds must be less than upper bounds"
        
        # Initialize state
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.personal_best_metadata = None  # Track metadata for personal bests
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.global_best_metadata = None  # Track metadata for global best
        
        # Multi-swarm state
        self.sub_swarms = None
        self.local_best_positions = None
        self.local_best_scores = None
        
        # Tracking
        self.history = {
            'best_scores': [],
            'mean_scores': [],
            'diversity': [],
            'parameters': [],
            'diversity_states': []
        }
        self.metadata_history = []  # Store metadata for each new best
        self.n_evaluations = 0
        self.initial_diversity = None
        self.supports_metadata = None  # Auto-detect on first evaluation
        self.print_metadata = print_metadata
        self.logger = logging.getLogger(__name__)
        if self.config.verbose:
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            self.logger.addHandler(ch)
        
    def optimize(self) -> PSOResult:
        """
        Run the optimization process.
        
        Returns:
            PSOResult with best position, score, metadata, and optimization history
        """
        self._print_header()
        
        # Initialize population
        self._initialize_population()
        
        # Setup multi-swarm if enabled
        if self.config.use_multi_swarm:
            self._initialize_multi_swarm()
        
        # Main optimization loop
        no_improvement_count = 0
        n_new_bests = 0
        convergence_iteration = None
        
        for iteration in range(self.config.max_iterations):
            self._print_iteration_header(iteration)
            
            # Compute and adapt parameters
            diversity, diversity_normalized = self._compute_diversity()
            diversity_state = self._classify_diversity_state(diversity_normalized)
            w, c1, c2 = self._get_pso_parameters(
                diversity_normalized, iteration
            )
            
            self.logger.info(f"Diversity: {diversity:.6f} (normalized: {diversity_normalized:.3f}, "
                             f"state: {diversity_state.value})")
            self.logger.info(f"PSO parameters: w={w:.3f}, c1={c1:.3f}, c2={c2:.3f}")
            
            # Update swarm
            self._update_velocities_and_positions(w, c1, c2)
            
            # Evaluate
            scores, metadata_list = self._evaluate_population()
            
            # Update personal and global bests
            improved, new_best_found, new_best_metadata = self._update_bests(
                scores, metadata_list
            )
            
            self.logger.info(f"Particles improved: {np.sum(improved)}/{self.config.n_particles}")
            self.logger.info(f"Score range: [{np.min(scores):.6f}, {np.max(scores):.6f}], "
                             f"mean: {np.mean(scores):.6f}")
            
            # Handle new best
            if new_best_found:
                improvement = self.history['best_scores'][-1] - self.global_best_score \
                             if self.history['best_scores'] else 0
                self.logger.info(f"\nNew best: {self.global_best_score:.6f} "
                                 f"(improvement: {improvement:.6f})")
                
                # Store metadata for this new best
                if self.config.track_metadata and new_best_metadata is not None:
                    self.metadata_history.append({
                        'iteration': iteration,
                        'score': self.global_best_score,
                        'position': self.global_best_position.copy(),
                        'metadata': new_best_metadata
                    })
                
                if n_new_bests % self.config.diagnostic_frequency == 0:
                    self._print_diagnostics()
                
                n_new_bests += 1
                no_improvement_count = 0
                
                # Track first convergence to acceptable threshold
                if convergence_iteration is None and self.global_best_score < 1e-6:
                    convergence_iteration = iteration
            else:
                no_improvement_count += 1
                self.logger.info(f"No improvement (count: {no_improvement_count})")
            
            # Update multi-swarm
            if self.config.use_multi_swarm:
                self._update_sub_swarm_bests()
                
                # Check for regrouping
                if (iteration + 1) % self.config.regrouping_period == 0:
                    self.logger.info(f"\nRegrouping sub-swarms...")
                    self._regroup_sub_swarms()
            
            # Handle stagnation
            if no_improvement_count >= self.config.obl_escape_threshold and \
               self.config.obl_escape_enabled:
                # Try OBL escape first
                self.logger.info(f"\nStagnation detected ({no_improvement_count} iterations)")
                self._obl_escape()
                no_improvement_count = 0  # Reset after escape
                
            elif no_improvement_count >= self.config.restart_threshold:
                # If still stagnating, do intelligent restart
                self.logger.info(f"\nRestarting...")
                self._intelligent_restart()
                no_improvement_count = 0  # Reset after restart
            
            # Record history
            self.history['best_scores'].append(self.global_best_score)
            self.history['mean_scores'].append(np.mean(scores))
            self.history['diversity'].append(diversity)
            self.history['diversity_states'].append(diversity_state.value)
            self.history['parameters'].append({
                'w': w, 'c1': c1, 'c2': c2,
                'diversity_normalized': diversity_normalized,
                'diversity_state': diversity_state.value
            })
            
            self.logger.info(f"\nBest score so far: {self.global_best_score:.6f}")
        
        # Finalize
        self._print_footer()
        
        return PSOResult(
            best_position=self.global_best_position.copy(),
            best_score=self.global_best_score,
            best_metadata=self.global_best_metadata,
            history=self.history,
            metadata_history=self.metadata_history if self.config.track_metadata else [],
            n_iterations=self.config.max_iterations,
            n_evaluations=self.n_evaluations,
            n_new_bests=n_new_bests,
            final_diversity=diversity,
            convergence_iteration=convergence_iteration,
            config=self.config
        )
    
    # ========================================================================
    # Initialization Methods
    # ========================================================================
    
    def _initialize_population(self):
        """Initialize swarm population with optional OBL."""
        if self.config.use_obl_initialization:
            self._initialize_with_obl()
        else:
            self._initialize_random()
        
        # Initialize velocities
        self.velocities = np.random.randn(
            self.config.n_particles, self.n_dimensions
        ) * 0.1 * (self.upper_bounds - self.lower_bounds)
        
        # Compute initial diversity
        self.initial_diversity, _ = self._compute_diversity()
        
        self.logger.info(f"Initial best score: {self.global_best_score:.6f}")
        self.logger.info(f"Initial diversity: {self.initial_diversity:.6f}")
        if self.supports_metadata:
            self.logger.info(f"Metadata tracking enabled")
    
    def _initialize_random(self):
        """Standard random initialization."""
        self.positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds,
            (self.config.n_particles, self.n_dimensions)
        )
        
        # Evaluate initial population
        scores, metadata_list = self._evaluate_population()
        
        # Set personal bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = scores
        if self.supports_metadata:
            self.personal_best_metadata = metadata_list
        
        # Set global best
        best_idx = np.argmin(scores)
        self.global_best_position = self.positions[best_idx].copy()
        self.global_best_score = scores[best_idx]
        if self.supports_metadata:
            self.global_best_metadata = metadata_list[best_idx]
    
    def _initialize_with_obl(self):
        """Initialize with Opposition-Based Learning."""
        self.logger.info("Initializing with Opposition-Based Learning...")
        
        # Generate random positions
        random_positions = np.random.uniform(
            self.lower_bounds, self.upper_bounds,
            (self.config.n_particles, self.n_dimensions)
        )
        
        # Calculate opposites
        opposite_positions = self._calculate_opposites(random_positions)
        
        # Combine and evaluate
        all_positions = np.vstack([random_positions, opposite_positions])
        result = self.objective_function(all_positions)
        
        # Handle both return types
        if isinstance(result, tuple):
            all_scores, all_metadata = result
            self.supports_metadata = True
        else:
            all_scores = result
            all_metadata = [None] * len(all_scores)
            self.supports_metadata = False
        
        self.n_evaluations += len(all_scores)
        
        # Select best n_particles
        best_indices = np.argsort(all_scores)[:self.config.n_particles]
        self.positions = all_positions[best_indices]
        scores = all_scores[best_indices]
        
        # Set personal bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = scores
        if self.supports_metadata:
            self.personal_best_metadata = [all_metadata[i] for i in best_indices]
        
        # Set global best
        self.global_best_position = self.positions[0].copy()
        self.global_best_score = scores[0]
        if self.supports_metadata:
            self.global_best_metadata = all_metadata[best_indices[0]]
        
        self.logger.info(f"  OBL: Best = {scores[0]:.6f}, Worst = {scores[-1]:.6f}")
    
    def _initialize_multi_swarm(self):
        """Initialize sub-swarms for DMS-PSO."""
        particles_per_swarm = self.config.n_particles // self.config.n_sub_swarms
        
        self.sub_swarms = []
        for i in range(self.config.n_sub_swarms):
            start_idx = i * particles_per_swarm
            end_idx = start_idx + particles_per_swarm
            if i == self.config.n_sub_swarms - 1:
                end_idx = self.config.n_particles
            self.sub_swarms.append(list(range(start_idx, end_idx)))
        
        self.local_best_positions = [None] * self.config.n_sub_swarms
        self.local_best_scores = [float('inf')] * self.config.n_sub_swarms
        
        self._update_sub_swarm_bests()
        
        self.logger.info(f"Initialized {self.config.n_sub_swarms} sub-swarms")
        for i, swarm in enumerate(self.sub_swarms):
            self.logger.info(f"  Sub-swarm {i}: {len(swarm)} particles, "
                             f"best score = {self.local_best_scores[i]:.6f}")

    
    # ========================================================================
    # Core PSO Update Methods
    # ========================================================================
    
    def _update_velocities_and_positions(self, w: float, c1: float, c2: float):
        """Update particle velocities and positions."""
        if self.config.use_multi_swarm:
            # Multi-swarm: use local best for each sub-swarm
            for swarm_id, particle_indices in enumerate(self.sub_swarms):
                for idx in particle_indices:
                    r1 = np.random.random(self.n_dimensions)
                    r2 = np.random.random(self.n_dimensions)
                    
                    cognitive = c1 * r1 * (
                        self.personal_best_positions[idx] - self.positions[idx]
                    )
                    social = c2 * r2 * (
                        self.local_best_positions[swarm_id] - self.positions[idx]
                    )
                    
                    self.velocities[idx] = w * self.velocities[idx] + cognitive + social
                    self.positions[idx] = self.positions[idx] + self.velocities[idx]
                    self.positions[idx] = np.clip(
                        self.positions[idx], self.lower_bounds, self.upper_bounds
                    )
        else:
            # Standard PSO: use global best
            for i in range(self.config.n_particles):
                r1 = np.random.random(self.n_dimensions)
                r2 = np.random.random(self.n_dimensions)
                
                cognitive = c1 * r1 * (
                    self.personal_best_positions[i] - self.positions[i]
                )
                social = c2 * r2 * (
                    self.global_best_position - self.positions[i]
                )
                
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(
                    self.positions[i], self.lower_bounds, self.upper_bounds
                )
    
    def _evaluate_population(self) -> Tuple[np.ndarray, List[Optional[Dict]]]:
        """
        Evaluate all particles and return scores and metadata.
        
        Returns:
            scores: np.ndarray of shape (n_particles,)
            metadata_list: List of metadata dicts (or None if not supported)
        """
        result = self.objective_function(self.positions)
        self.n_evaluations += self.config.n_particles
        
        # Handle both return types
        if isinstance(result, tuple):
            scores, metadata_list = result
            if self.supports_metadata is None:
                self.supports_metadata = True
        else:
            scores = result
            metadata_list = [None] * len(scores)
            if self.supports_metadata is None:
                self.supports_metadata = False
        
        return scores, metadata_list
    
    def _update_bests(self, scores: np.ndarray, 
                     metadata_list: List[Optional[Dict]]) -> Tuple[np.ndarray, bool, Optional[Dict]]:
        """
        Update personal and global bests.
        
        Returns:
            improved: Boolean array indicating which particles improved
            new_best_found: Whether a new global best was found
            new_best_metadata: Metadata for new global best (if found)
        """
        # Update personal bests
        improved = scores < self.personal_best_scores
        self.personal_best_scores[improved] = scores[improved]
        self.personal_best_positions[improved] = self.positions[improved]
        
        # Update personal best metadata
        if self.supports_metadata and self.personal_best_metadata is not None:
            for i, (is_improved, metadata) in enumerate(zip(improved, metadata_list)):
                if is_improved:
                    self.personal_best_metadata[i] = metadata
        
        # Update global best
        min_idx = np.argmin(scores)
        new_best_found = False
        new_best_metadata = None
        
        if scores[min_idx] < self.global_best_score:
            self.global_best_score = scores[min_idx]
            self.global_best_position = self.positions[min_idx].copy()
            new_best_found = True
            
            if self.supports_metadata:
                self.global_best_metadata = metadata_list[min_idx]
                new_best_metadata = metadata_list[min_idx]
        
        return improved, new_best_found, new_best_metadata
    
    # ========================================================================
    # Diversity and Adaptive Parameters
    # ========================================================================
    
    def _compute_diversity(self) -> Tuple[float, float]:
        """
        Compute swarm diversity.
        
        Returns:
            diversity: Raw diversity (mean std across dimensions)
            diversity_normalized: Normalized by initial diversity
        """
        diversity = np.mean(np.std(self.positions, axis=0))
        
        if self.initial_diversity is not None and self.initial_diversity > 1e-10:
            diversity_normalized = diversity / self.initial_diversity
        else:
            diversity_normalized = 1.0
        
        return diversity, diversity_normalized
    
    def _classify_diversity_state(self, diversity_normalized: float) -> DiversityState:
        """Classify diversity into low/medium/high state."""
        if diversity_normalized < self.config.diversity_low_threshold:
            return DiversityState.LOW
        elif diversity_normalized > self.config.diversity_high_threshold:
            return DiversityState.HIGH
        else:
            return DiversityState.MEDIUM
    
    def _get_pso_parameters(self, diversity_normalized: float, 
                           iteration: int) -> Tuple[float, float, float]:
        """
        Get PSO parameters (w, c1, c2) based on diversity and iteration.
        
        Returns:
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        if not self.config.use_adaptive_parameters:
            # Standard schedule
            progress = iteration / self.config.max_iterations
            w = self.config.w_max - (self.config.w_max - self.config.w_min) * progress
            c1 = np.random.uniform(*self.config.c1_range)
            c2 = np.random.uniform(*self.config.c2_range)
            return w, c1, c2
        
        # Adaptive parameters
        progress = iteration / self.config.max_iterations
        w_baseline = self.config.w_max - \
                    (self.config.w_max - self.config.w_min) * progress
        
        low_thresh = self.config.diversity_low_threshold
        high_thresh = self.config.diversity_high_threshold
        
        if diversity_normalized < low_thresh:
            # Low diversity - exploration mode
            w = min(self.config.w_max, w_baseline + 0.2)
            c1 = 1.2  # Lower cognitive (don't trust personal best too much)
            c2 = 2.5  # Higher social (follow swarm to disperse)
        elif diversity_normalized > high_thresh:
            # High diversity - exploitation mode
            w = max(self.config.w_min, w_baseline - 0.1)
            c1 = 2.5  # Higher cognitive (trust personal experience)
            c2 = 1.5  # Lower social (less swarm influence)
        else:
            # Medium diversity - balanced
            w = w_baseline
            # Interpolate between exploration and exploitation
            diversity_factor = (diversity_normalized - low_thresh) / \
                             (high_thresh - low_thresh)
            c1 = 1.2 + 1.3 * diversity_factor  # 1.2 -> 2.5
            c2 = 2.5 - 1.0 * diversity_factor  # 2.5 -> 1.5
        
        return w, c1, c2
    
    # ========================================================================
    # Multi-Swarm Methods
    # ========================================================================
    
    def _update_sub_swarm_bests(self):
        """Update local best for each sub-swarm."""
        for swarm_id, particle_indices in enumerate(self.sub_swarms):
            swarm_scores = self.personal_best_scores[particle_indices]
            best_idx_in_swarm = np.argmin(swarm_scores)
            best_particle_idx = particle_indices[best_idx_in_swarm]
            
            self.local_best_scores[swarm_id] = \
                self.personal_best_scores[best_particle_idx]
            self.local_best_positions[swarm_id] = \
                self.personal_best_positions[best_particle_idx].copy()
    
    def _regroup_sub_swarms(self):
        """Randomly regroup particles into new sub-swarms."""
        particles_per_swarm = self.config.n_particles // self.config.n_sub_swarms
        shuffled_indices = np.random.permutation(self.config.n_particles)
        
        self.sub_swarms = []
        for i in range(self.config.n_sub_swarms):
            start_idx = i * particles_per_swarm
            end_idx = start_idx + particles_per_swarm
            if i == self.config.n_sub_swarms - 1:
                end_idx = self.config.n_particles
            self.sub_swarms.append(shuffled_indices[start_idx:end_idx].tolist())
        
        self._update_sub_swarm_bests()
    
    # ========================================================================
    # Stagnation Handling
    # ========================================================================
    
    def _obl_escape(self):
        """Apply opposition-based escape for worst particles."""
        n_worst = max(1, int(self.config.n_particles * 
                            self.config.obl_escape_fraction))
        worst_indices = np.argsort(self.personal_best_scores)[-n_worst:]
        
        self.logger.info(f"  OBL escape: Replacing {n_worst} worst particles")
        
        for idx in worst_indices:
            opposite = self._calculate_opposites(
                self.positions[idx:idx+1]
            )[0]
            self.positions[idx] = opposite
            # Reset velocity for escaped particles
            self.velocities[idx] = np.random.randn(self.n_dimensions) * 0.1 * \
                                  (self.upper_bounds - self.lower_bounds)
    
    def _intelligent_restart(self):
        """Perform intelligent restart keeping elites."""
        n_elite = max(1, int(self.config.n_particles * 
                            self.config.elite_fraction))
        elite_indices = np.argsort(self.personal_best_scores)[:n_elite]
        
        self.logger.info(f"  Keeping {n_elite} elites, reinitializing "
                         f"{self.config.n_particles - n_elite} particles")
        
        for i in range(self.config.n_particles):
            if i not in elite_indices:
                # Select random elite to perturb around
                elite_idx = elite_indices[np.random.randint(n_elite)]
                
                # Gaussian perturbation
                std = self.config.restart_std_factor * \
                      (self.upper_bounds - self.lower_bounds)
                noise = np.random.randn(self.n_dimensions) * std
                
                self.positions[i] = self.personal_best_positions[elite_idx] + noise
                self.positions[i] = np.clip(
                    self.positions[i], self.lower_bounds, self.upper_bounds
                )
                
                # Reset velocity
                self.velocities[i] = np.random.randn(self.n_dimensions) * 0.1 * \
                                    (self.upper_bounds - self.lower_bounds)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _calculate_opposites(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate opposite positions in search space.
        
        For position x in [a, b], opposite is: x_opp = a + b - x
        """
        opposites = self.lower_bounds + self.upper_bounds - positions
        return np.clip(opposites, self.lower_bounds, self.upper_bounds)
    
    # ========================================================================
    # Printing Methods
    # ========================================================================
    
    def _print_header(self):
        """Print optimization header."""
        if not self.config.verbose:
            return

        subswarm_info = ""
        if self.config.use_multi_swarm:
            subswarm_info = f"    Sub-swarms: {self.config.n_sub_swarms}\n" \
                f"    Regrouping: Every {self.config.regrouping_period} iterations"
            
        info = ("\n" + "="*80,
                "Adaptive Particle Swarm Optimization",
                "="*80,
                f"Configuration:",
                f"  Particles: {self.config.n_particles}",
                f"  Iterations: {self.config.max_iterations}",
                f"  Dimensions: {self.n_dimensions}",
                f"  OBL Initialization: {self.config.use_obl_initialization}",
                f"  Adaptive Parameters: {self.config.use_adaptive_parameters}",
                f"  Multi-Swarm: {self.config.use_multi_swarm}",
                subswarm_info,
                f"  Metadata Tracking: {self.config.track_metadata}",
                "="*80 + "\n")
        
        for item in info:
            self.logger.info(item)
    
    def _print_iteration_header(self, iteration: int):
        """Print iteration header."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Iteration {iteration+1}/{self.config.max_iterations}")
        self.logger.info(f"{'='*80}")
    
    def _print_diagnostics(self):
        """Print detailed diagnostics."""
        if not self.config.verbose:
            return
        
        self.logger.info(f"\n{'#'*80}")
        self.logger.info("Best solution diagnostics")
        self.logger.info(f"{'#'*80}")
        self.logger.info(f"Score: {self.global_best_score:.6f}")
        self.logger.info(f"Position (first 10 dims):")
        for i, val in enumerate(self.global_best_position[:10]):
            self.logger.info(f"  Dim {i}: {val:.6f}")
        if self.n_dimensions > 10:
            self.logger.info(f"  ... and {self.n_dimensions - 10} more dimensions")
        
        if self.supports_metadata and self.global_best_metadata is not None:
            self.logger.info(f"\nMetadata available for this configuration:")
            if self.print_metadata is None:
                self.logger.info(pprint.pformat(self.global_best_metadata))
            else:
                self.print_metadata(self.global_best_metadata, logger=self.logger)
            
        self.logger.info(f"{'#'*80}")
    
    def _print_footer(self):
        """Print optimization completion."""
        if not self.config.verbose:
            return
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Optimization complete")
        self.logger.info("="*80)
        self.logger.info(f"Best score: {self.global_best_score:.6f}")
        self.logger.info(f"Total evaluations: {self.n_evaluations}")
        self.logger.info(f"Function evals per iteration: "
              f"{self.n_evaluations / self.config.max_iterations:.1f}")
        if self.config.track_metadata:
            self.logger.info(f"Metadata snapshots stored: {len(self.metadata_history)}")
        self.logger.info("="*80)


# ============================================================================
# Benchmark Functions for Testing
# ============================================================================

class BenchmarkFunctions:
    """Collection of standard benchmark functions for testing PSO."""
    
    @staticmethod
    def sphere(x: np.ndarray) -> np.ndarray:
        """
        Sphere function: sum of squares.
        Global minimum: f(0,...,0) = 0
        Unimodal, convex, separable.
        """
        return np.sum(x**2, axis=1)
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> np.ndarray:
        """
        Rosenbrock function (banana function).
        Global minimum: f(1,...,1) = 0
        Unimodal, non-convex, non-separable.
        Valley is easy to find but minimum is hard.
        """
        return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + 
                     (1 - x[:, :-1])**2, axis=1)
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> np.ndarray:
        """
        Rastrigin function.
        Global minimum: f(0,...,0) = 0
        Highly multimodal, separable.
        Many local minima.
        """
        n = x.shape[1]
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
    
    @staticmethod
    def ackley(x: np.ndarray) -> np.ndarray:
        """
        Ackley function.
        Global minimum: f(0,...,0) = 0
        Highly multimodal, non-separable.
        Nearly flat outer region.
        """
        n = x.shape[1]
        sum_sq = np.sum(x**2, axis=1)
        sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
        return (-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - 
                np.exp(sum_cos / n) + 20 + np.e)
    
    @staticmethod
    def griewank(x: np.ndarray) -> np.ndarray:
        """
        Griewank function.
        Global minimum: f(0,...,0) = 0
        Multimodal, non-separable.
        Many widespread local minima.
        """
        sum_sq = np.sum(x**2, axis=1)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
        return sum_sq / 4000 - prod_cos + 1
