import numpy as np
import torch
import torch.multiprocessing as mp

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.problem import StarmapParallelization
from multiprocessing import get_context

from src.cuda_quantum_ops import GPUQuantumOps
from src.cuda_helpers import GPUDeviceWrapper
from src.qutip_quantum_ops import operator_new, p0_projector
from src.logger import setup_logger
from qutip import expect, displace, squeeze, basis, Qobj

# Global variable for worker processes
GLOBAL_PROBLEM = None

def set_gpu_device_silent(device_id):
    """
    Set the GPU device without printing any messages.
    
    Args:
        device_id (int): The ID of the GPU device to use.
        
    Raises:
        ValueError: If the specified device_id is not available.
        RuntimeError: If no CUDA-capable GPU devices are found.
    """
    if torch.cuda.is_available():
        if device_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU device {device_id} not found. Available devices: 0-{torch.cuda.device_count()-1}"
            )
        torch.cuda.set_device(device_id)
    else:
        raise RuntimeError("No CUDA-capable GPU devices found")

def init_worker(N, u, c, k, parity, projector, device_id):
    """
    Pool initializer function for multiprocessing workers.
    
    Args:
        N (int): Dimension of the Fock space.
        u (float): Parameter for quantum operations.
        c (float): Parameter for quantum operations.
        k (int): Parameter for quantum operations.
        parity (str): Parity of the quantum state ('even' or 'odd').
        projector (Qobj): QuTiP projector object.
        device_id (int): GPU device ID to use.
    """
    global GLOBAL_PROBLEM
    set_gpu_device_silent(device_id)
    GLOBAL_PROBLEM = GPUQuantumParetoProblem(
        N, u, c, k, parity, projector=projector, device_id=device_id, parallelization=None
    )

def evaluate_chunk(chunk):
    """
    Top-level function to evaluate a chunk of candidate solutions.
    Uses the global GPUQuantumParetoProblem instance.
    
    Args:
        chunk (numpy.ndarray): Array of candidate solutions to evaluate.
        
    Returns:
        numpy.ndarray: Array of objective function values for each candidate.
    """
    global GLOBAL_PROBLEM
    set_gpu_device_silent(GLOBAL_PROBLEM.device_id)
    out_chunk = {"F": None}
    GLOBAL_PROBLEM._evaluate(chunk, out_chunk)
    return out_chunk["F"]

logger = setup_logger()

class GPUQuantumParetoProblem(Problem):
    """
    Multi-objective optimization problem for quantum state optimization using GPU.
    This class implements a parallel version of the quantum state optimization
    problem using GPU acceleration.

    Attributes:
        N (int): Dimension of the Fock space.
        u (float): Parameter for quantum operations.
        c (float): Parameter for quantum operations.
        k (int): Parameter for quantum operations.
        device_id (int): GPU device ID to use.
        device (str): CUDA device string.
        quantum_ops (GPUDeviceWrapper): GPU-accelerated quantum operations handler.
        parity (str): Parity of the quantum state ('even' or 'odd').
        squeezing_bound (float): Bound used to decide if a candidate is valid.
        operator (torch.Tensor): Precomputed operator used in evaluation.
        projector (torch.Tensor): Projector tensor used for fidelity calculations.
    """

    def __init__(self, N, u, c, k, parity, projector=None, device_id=0, parallelization=None):
        """
        Initialize the GPU quantum pareto optimization problem.
        
        Args:
            N (int): Dimension of the Fock space.
            u (float): Parameter for quantum operations.
            c (float): Parameter for quantum operations.
            k (int): Parameter for quantum operations.
            parity (str): Parity of the quantum state ('even' or 'odd').
            projector (Qobj, optional): QuTiP projector object. Defaults to None.
            device_id (int, optional): GPU device ID to use. Defaults to 0.
            parallelization (StarmapParallelization, optional): Parallelization strategy. Defaults to None.
            
        Raises:
            ValueError: If parity is not 'even' or 'odd', or if projector is invalid.
        """
        set_gpu_device_silent(device_id)
        self.N = N
        self.u = u
        self.c = c
        self.k = k
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        self.quantum_ops = GPUDeviceWrapper(GPUQuantumOps(N), device_id)
        self.parity = parity
        self.squeezing_bound = np.real(
            expect(
                operator_new(self.N, self.u, 0.0, self.c, self.k),
                operator_new(self.N, self.u, np.pi, self.c, self.k).groundstate()[1],
            )
        )
        if parity == "even":
            self.operator = self.quantum_ops.operator_new_gpu(self.u, 0, self.c, self.k)
        elif parity == "odd":
            self.operator = self.quantum_ops.operator_new_gpu(self.u, np.pi, self.c, self.k)
        else:
            error_msg = f"Parity must be either 'even' or 'odd', not '{parity}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if projector is None:
            self.projector = torch.tensor(
                p0_projector(self.N).full(), dtype=torch.complex64
            ).to(self.device)
        elif isinstance(projector, Qobj):
            self.projector = torch.tensor(projector.full(), dtype=torch.complex64).to(self.device)
        else:
            error_msg = "Projector must be a QuTiP Qobj"
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_var = 2 * N
        bound = 10

        logger.info(
            f"Initializing GPUQuantumParetoProblem with N={N}, u={u}, c={c}, k={k}, device_id={device_id}"
        )
        logger.debug(f"Squeezing bound calculated: {self.squeezing_bound}")

        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=-bound, xu=bound, parallelization=parallelization)

    def _check_memory(self):
        """
        Check available GPU memory and calculate a safe batch size for processing.
        
        Returns:
            int: Safe batch size based on available GPU memory.
            
        Note:
            The batch size is calculated based on the size of complex64 elements
            and ensures at least one sample can be processed.
        """
        total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device_id)
        free_memory = total_memory - allocated_memory

        element_size = 4 * self.N * 2
        safe_batch_size = free_memory // element_size

        logger.debug(
            f"Memory check: total={total_memory/1e9:.2f}GB, allocated={allocated_memory/1e9:.2f}GB, "
            f"free={free_memory/1e9:.2f}GB, safe_batch_size={max(1, safe_batch_size)}"
        )
        return max(1, safe_batch_size)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions for a batch of candidate solutions.
        
        Args:
            x (numpy.ndarray): Array of candidate solutions to evaluate.
            out (dict): Dictionary to store evaluation results with key "F".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Note:
            The evaluation computes two objectives:
            1. Squeezing value (to be minimized)
            2. Output fidelity (to be minimized)
            
            If a candidate exceeds the squeezing bound or produces invalid results,
            it is assigned worst-case values (1e8 for squeezing, 1.0 for fidelity).
        """
        try:
            x_gpu = torch.tensor(x, dtype=torch.complex64).to(self.device)
            f = torch.zeros((x_gpu.shape[0], 2), device=self.device)
            optimal_batch_size = self._check_memory()
            logger.debug(f"Evaluating {len(x_gpu)} candidates with batch size {optimal_batch_size}")

            for i in range(0, len(x_gpu), optimal_batch_size):
                batch = x_gpu[i : i + optimal_batch_size]
                states = self.params_to_state_gpu(batch)

                for j, state in enumerate(states):
                    try:
                        cat_sq, out_fid = self.quantum_ops.catfid_gpu(
                            state, self.u, self.parity, self.c, self.k, self.projector, self.operator
                        )

                        if torch.isnan(cat_sq) or torch.isinf(cat_sq):
                            cat_sq = torch.tensor(float("inf"), device=self.device)
                            logger.warning("NaN or Inf detected in cat_sq, using infinity")
                        if torch.isnan(out_fid) or torch.isinf(out_fid):
                            out_fid = torch.tensor(1.0, device=self.device)
                            logger.warning("NaN or Inf detected in out_fid, using 1.0")

                        if np.real(cat_sq.item()) > self.squeezing_bound:
                            f[i + j, 0] = torch.tensor(1e8, device=self.device)
                            f[i + j, 1] = torch.tensor(1.0, device=self.device)
                            logger.debug(f"Candidate {i+j} exceeds squeezing bound, assigning worst values")
                        else:
                            f[i + j, 0] = torch.abs(cat_sq)
                            f[i + j, 1] = torch.abs(out_fid)
                    except Exception as e:
                        logger.error(f"Error in evaluation: {str(e)}")
                        f[i + j, 0] = torch.tensor(float("inf"), device=self.device)
                        f[i + j, 1] = torch.tensor(0, device=self.device)
            out["F"] = f.cpu().numpy()
        except Exception as e:
            logger.error(f"Fatal error in evaluation: {str(e)}")
            out["F"] = np.full((x.shape[0], 2), np.inf)

    def params_to_state_gpu(self, params):
        """
        Convert optimization parameters to normalized quantum states on GPU.
        
        Args:
            params (torch.Tensor): Input parameters tensor containing real and imaginary parts.
            
        Returns:
            torch.Tensor: Normalized quantum states tensor.
            
        Note:
            The function handles invalid values (NaN, Inf) by replacing them with
            finite values (LARGE_NUM=1e8 for Inf, SMALL_NUM=1e-14 for NaN).
            The states are normalized to ensure they represent valid quantum states.
        """
        LARGE_NUM = 1e8
        SMALL_NUM = 1e-14
        try:
            params = params.real if torch.is_complex(params) else params
            if torch.any(torch.isnan(params)) or torch.any(torch.isinf(params)):
                params = params.clone()
                inf_mask = torch.isinf(params)
                nan_mask = torch.isnan(params)
                inf_count = torch.sum(inf_mask).item()
                nan_count = torch.sum(nan_mask).item()
                if inf_count > 0 or nan_count > 0:
                    logger.warning(f"Found {inf_count} inf and {nan_count} NaN values in params, replacing them")
                params[inf_mask] = torch.sign(params[inf_mask]) * LARGE_NUM
                params[nan_mask] = torch.rand_like(params[nan_mask]) * SMALL_NUM

            real_parts = params[:, : self.N]
            imag_parts = params[:, self.N :]
            states = torch.complex(real_parts.float(), imag_parts.float())
            abs_squared = torch.abs(states) ** 2
            abs_squared = torch.where(torch.isnan(abs_squared), torch.ones_like(abs_squared) * SMALL_NUM, abs_squared)
            abs_squared = torch.where(torch.isinf(abs_squared), torch.ones_like(abs_squared) * LARGE_NUM, abs_squared)
            norms = torch.sqrt(torch.sum(abs_squared, dim=1, keepdim=True))
            norms = torch.where((norms == 0) | torch.isnan(norms), torch.ones_like(norms) * SMALL_NUM, norms)
            norms = torch.where(torch.isinf(norms), torch.ones_like(norms) * torch.sqrt(torch.tensor(LARGE_NUM)), norms)
            states = states / norms
            if torch.any(torch.isnan(states)) or torch.any(torch.isinf(states)):
                invalid_mask = torch.any(torch.isnan(states) | torch.isinf(states), dim=1, keepdim=True)
                invalid_count = torch.sum(invalid_mask).item()
                logger.warning(f"Found {invalid_count} invalid states after normalization, using fallback states")
                fallback_states = torch.complex(torch.ones_like(states.real) * torch.sqrt(LARGE_NUM / self.N),
                                             torch.zeros_like(states.imag))
                states = torch.where(invalid_mask, fallback_states, states)
            return states
        except Exception as e:
            logger.error(f"Error in params_to_state_gpu: {str(e)}")
            batch_size = params.shape[0]
            fallback = torch.complex(
                torch.ones((batch_size, self.N), device=params.device) * torch.sqrt(LARGE_NUM / self.N),
                torch.zeros((batch_size, self.N), device=params.device),
            )
            return fallback


class OptimizationCallback(Callback):
    """
    Callback to record and log optimization progress during NSGA-II execution.
    
    Attributes:
        data (dict): Dictionary storing optimization history with key "F" for objective values.
        logger (Logger): Logger instance for recording progress.
    """

    def __init__(self) -> None:
        """Initialize the callback with empty data storage."""
        super().__init__()
        self.data["F"] = []
        self.logger = setup_logger()
        self.logger.info("Optimization callback initialized")

    def notify(self, algorithm):
        """
        Record optimization progress at each generation.
        
        Args:
            algorithm: The NSGA-II algorithm instance.
            
        Note:
            Logs the minimum values of both objective functions and population size
            at each generation.
        """
        generation = algorithm.n_gen
        if hasattr(algorithm, "opt") and algorithm.opt is not None:
            opt_F = algorithm.opt.get("F")
            if opt_F is not None:
                self.data["F"].append(opt_F.copy())
                min_f1 = opt_F[:, 0].min() if len(opt_F) > 0 else float("inf")
                min_f2 = opt_F[:, 1].min() if len(opt_F) > 0 else float("inf")
                self.logger.info(
                    f"Generation {generation}: Min SQE={min_f1:.6f}, Min Fidelity={min_f2:.6f}, "
                    f"Population size={len(opt_F)}"
                )
            else:
                self.logger.warning(f"Generation {generation}: No optimal solutions found yet")
        else:
            self.logger.warning(f"Generation {generation}: Algorithm optimization data not available")


def optimize_quantum_state_gpu_cpu(
    N, u, c, k, projector=None, initial_kets=None, pop_size=500,
    max_generations=2000, num_workers=12, verbose=True, tolerance=5e-4,
    parity="even", device_id=0,
):
    """
    Perform a parallel GPU-CPU hybrid optimization of quantum states using NSGA-II.
    
    Args:
        N (int): Dimension of the Fock space.
        u (float): Parameter for quantum operations.
        c (float): Parameter for quantum operations.
        k (int): Parameter for quantum operations.
        projector (Qobj, optional): QuTiP projector object. Defaults to None.
        initial_kets (list, optional): List of initial quantum states. Defaults to None.
        pop_size (int, optional): Population size for NSGA-II. Defaults to 500.
        max_generations (int, optional): Maximum number of generations. Defaults to 2000.
        num_workers (int, optional): Number of CPU workers for parallel evaluation. Defaults to 12.
        verbose (bool, optional): Whether to print progress information. Defaults to True.
        tolerance (float, optional): Convergence tolerance. Defaults to 5e-4.
        parity (str, optional): Parity of the quantum state ('even' or 'odd'). Defaults to "even".
        device_id (int, optional): GPU device ID to use. Defaults to 0.
        
    Returns:
        Result: Optimization result object containing the Pareto front and other information.
        
    Raises:
        RuntimeError: If no CUDA-capable GPU devices are found.
        ValueError: If the specified GPU device is not available.
        
    Note:
        This function implements a hybrid optimization strategy where:
        1. GPU is used for quantum state operations and objective function evaluation
        2. CPU workers are used for parallel evaluation of different candidates
        3. The optimization uses NSGA-II with SBX crossover and polynomial mutation
    """
    logger = setup_logger()
    if not torch.cuda.is_available():
        error_msg = "No CUDA-capable GPU devices found"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    if device_id >= torch.cuda.device_count():
        error_msg = f"GPU device {device_id} not found. Available devices: 0-{torch.cuda.device_count()-1}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    set_gpu_device_silent(device_id)
    if verbose:
        logger.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")

    if initial_kets is None:
        logger.info("No initial kets provided, generating candidates")
        initial_kets = [
            operator_new(N, u, 0, c, k).groundstate()[1],
            operator_new(N, u, np.pi, c, k).groundstate()[1],
        ]
        cats = [
            ((displace(N, u / np.sqrt(2)) + c * displace(N, -u / np.sqrt(2)))
             * squeeze(N, 1) * basis(N, 0)).unit()
            for c in range(-1000, 1001)
        ]
        initial_kets.extend(cats)
        logger.info(f"Generated {len(initial_kets)} initial candidate states")
        
    initial_states = []
    for i, initial_ket in enumerate(initial_kets):
        try:
            if not isinstance(initial_ket, Qobj) or not initial_ket.isket:
                error_msg = f"Initial ket {i} is not a valid QuTiP ket state"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if initial_ket.dims[0][0] != N:
                error_msg = f"Initial ket {i} dimension ({initial_ket.dims[0][0]}) does not match N ({N})"
                logger.error(error_msg)
                raise ValueError(error_msg)
            amplitudes = initial_ket.full().flatten()
            initial_state = np.concatenate([amplitudes.real, amplitudes.imag])
            initial_states.append(initial_state)
        except Exception as e:
            logger.error(f"Error processing initial ket {i}: {str(e)}")
            raise

    logger.info("Creating optimization problem instance")
    if num_workers is None:
        num_workers = mp.cpu_count()
    logger.info(f"Using {num_workers} CPU processes for parallel evaluation")

    ctx = get_context("spawn")
    pool = ctx.Pool(processes=num_workers, initializer=init_worker,
                    initargs=(N, u, c, k, parity, projector, device_id))

    def parallel_evaluation(X, return_values_of="auto", **kwargs):
        """
        Parallel evaluation function that splits work across CPU workers.
        
        Args:
            X (numpy.ndarray): Array of candidates to evaluate.
            return_values_of (str, optional): Type of values to return. Defaults to "auto".
            **kwargs: Additional keyword arguments.
            
        Returns:
            dict: Dictionary containing evaluation results with key "F".
        """
        chunk_size = max(1, X.shape[0] // num_workers)
        chunks = [X[i : i + chunk_size] for i in range(0, X.shape[0], chunk_size)]
        logger.debug(f"Splitting {X.shape[0]} candidates into {len(chunks)} chunks for parallel evaluation")
        results = pool.map(evaluate_chunk, chunks)
        return {"F": np.vstack(results)}

    problem = GPUQuantumParetoProblem(N, u, c, k, parity, projector, device_id, parallelization=None)
    problem.evaluate = parallel_evaluation

    class MultiInitialStateSampling(FloatRandomSampling):
        """
        Custom sampling operator that combines initial states with random samples.
        """
        def _do(self, problem, n_samples, **kwargs):
            """
            Perform sampling by combining initial states with random samples if needed.
            
            Args:
                problem: The optimization problem instance.
                n_samples (int): Number of samples required.
                **kwargs: Additional keyword arguments.
                
            Returns:
                numpy.ndarray: Array of sampled states.
            """
            if n_samples <= len(initial_states):
                logger.debug(f"Using {n_samples} initial states for sampling")
                return np.vstack(initial_states[:n_samples])
            logger.debug(f"Using {len(initial_states)} initial states and {n_samples - len(initial_states)} random samples")
            random_samples = super()._do(problem, n_samples - len(initial_states), **kwargs)
            initial_states_array = np.vstack(initial_states)
            return np.vstack([initial_states_array, random_samples])

    callback = OptimizationCallback()

    termination = DefaultMultiObjectiveTermination(
        xtol=tolerance, cvtol=tolerance, ftol=tolerance, period=50,
        n_max_gen=max_generations, n_max_evals=max_generations * pop_size,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MultiInitialStateSampling(),
        crossover=SBX(prob=0.9, eta=8),
        mutation=PM(prob=0.45, eta=6),
        eliminate_duplicates=True,
    )

    if verbose:
        logger.info("\nOptimization Settings:")
        logger.info(f"Population size: {pop_size}")
        logger.info(f"Maximum generations: {max_generations}")
        logger.info(f"Number of workers: {num_workers}")
        logger.info(f"GPU Device: {device_id} ({torch.cuda.get_device_name(device_id)})")
        logger.info("\nTermination Criteria:")
        logger.info(f"xtol: {termination.x.termination.tol}")
        logger.info(f"cvtol: {termination.cv.termination.tol}")
        logger.info(f"ftol: {termination.f.termination.tol}")
        logger.info(f"n_max_gen: {termination.max_gen.n_max_gen}\n")

    logger.info("Starting optimization")
    res = minimize(problem, algorithm, termination, seed=42, callback=callback, verbose=verbose)
    logger.info("Optimization completed")

    pool.close()
    pool.join()
    logger.info("Process pool closed")

    if parity == "even":
        out_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}"
    elif parity == "odd":
        out_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}_odd"

    logger.info(f"Creating output directory: {out_dir}")
    from src.helpers import create_optimization_animation
    animation_path = f"{out_dir}/pareto_front"
    logger.info(f"Creating optimization animation at {animation_path}")
    create_optimization_animation(callback.data["F"], animation_path)

    return res


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    logger.info("Starting main program")
    
    if torch.cuda.is_available():
        device_id = 0
        set_gpu_device_silent(device_id)
        logger.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        error_msg = "No CUDA-capable GPU devices found"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        
    try:
        logger.info("Starting optimization with default parameters")
        res = optimize_quantum_state_gpu_cpu(
            N=20, u=3, c=10, k=100, pop_size=500, max_generations=5000,
            verbose=True, device_id=device_id,
        )
        logger.info("Optimization completed successfully")
    except Exception as e:
        logger.error(f"Error in main program: {str(e)}", exc_info=True)
        raise
