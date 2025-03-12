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


from src.cuda_quantum_ops import GPUQuantumOps
from src.cuda_helpers import GPUDeviceWrapper
from src.qutip_quantum_ops import operator_new, p0_projector
from src.logging import setup_logger
from qutip import expect, displace, squeeze, basis, Qobj


# Set up the logger
logger = setup_logger()


class GPUQuantumParetoProblem(Problem):
    """
    Multi-objective optimization problem for quantum state optimization using GPU.

    Attributes:
        N (int): Dimension of the Fock space.
        u (float): Parameter for quantum operations.
        c (float): Parameter for quantum operations.
        k (int): Parameter for quantum operations.
        quantum_ops (GPUQuantumOps): GPU-accelerated quantum operations handler.
        squeezing_bound (float): Bound used to decide if a candidate is valid.
        operator: Precomputed operator used in evaluation.
        projector (torch.Tensor): Projector tensor used for fidelity calculations.
    """

    def __init__(self, N, u, c, k, parity, projector=None, device_id=0):
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
            self.operator = self.quantum_ops.operator_new_gpu(
                self.u, np.pi, self.c, self.k
            )
        else:
            error_msg = f"Parity must be either 'even' or 'odd', not '{parity}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if projector is None:
            self.projector = torch.tensor(
                p0_projector(self.N).full(), dtype=torch.complex64
            ).to(self.device)
        elif isinstance(projector, Qobj):
            self.projector = torch.tensor(projector.full(), dtype=torch.complex64).to(
                self.device
            )
        else:
            error_msg = "Projector must be a QuTiP Qobj"
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_var = 2 * N  # Real and imaginary parts
        bound = 10

        logger.info(f"Initializing GPUQuantumParetoProblem with N={N}, u={u}, c={c}, k={k}, device_id={device_id}")
        logger.debug(f"Squeezing bound calculated: {self.squeezing_bound}")

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=-bound,
            xu=bound,
        )

    def _check_memory(self):
        """
        Check available GPU memory and adjust batch size accordingly.

        Returns:
            int: Safe batch size based on available GPU memory.
        """
        total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device_id)
        free_memory = total_memory - allocated_memory

        element_size = (
            4 * self.N * 2
        )  # Approximate size in bytes for complex64 elements
        safe_batch_size = free_memory // element_size

        logger.debug(
            f"Memory check: total={total_memory/1e9:.2f}GB, "
            f"allocated={allocated_memory/1e9:.2f}GB, free={free_memory/1e9:.2f}GB, "
            f"safe_batch_size={max(1, safe_batch_size)}"
        )

        return max(1, safe_batch_size)  # Ensure at least a batch size of 1

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions for a given set of candidate solutions.

        Args:
            x (np.ndarray): Array of candidate solutions.
            out (dict): Dictionary to store evaluation results with key "F".
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
                            state,
                            self.u,
                            self.parity,
                            self.c,
                            self.k,
                            self.projector,
                            self.operator,
                        )

                        # Ensure valid numbers.
                        if torch.isnan(cat_sq) or torch.isinf(cat_sq):
                            cat_sq = torch.tensor(float("inf"), device=self.device)
                            logger.warning(f"NaN or Inf detected in cat_sq, using infinity")
                        if torch.isnan(out_fid) or torch.isinf(out_fid):
                            out_fid = torch.tensor(1.0, device=self.device)
                            logger.warning(f"NaN or Inf detected in out_fid, using 1.0")

                        # If candidate exceeds squeezing bound, assign worst values.
                        if np.real(cat_sq.item()) > self.squeezing_bound:
                            f[i + j, 0] = torch.tensor(1e8, device=self.device)
                            f[i + j, 1] = torch.tensor(1.0, device=self.device)
                            logger.debug(f"Candidate {i+j} exceeds squeezing bound, assigning worst values")
                        else:
                            f[i + j, 0] = torch.abs(cat_sq)
                            f[i + j, 1] = torch.abs(out_fid)
                    except Exception as e:
                        error_msg = f"Error in evaluation: {str(e)}"
                        logger.error(error_msg)
                        f[i + j, 0] = torch.tensor(float("inf"), device=self.device)
                        f[i + j, 1] = torch.tensor(0, device=self.device)

            out["F"] = f.cpu().numpy()
        except Exception as e:
            error_msg = f"Fatal error in evaluation: {str(e)}"
            logger.error(error_msg)
            out["F"] = np.full((x.shape[0], 2), np.inf)

    def params_to_state_gpu(self, params):
        """
        GPU version of params_to_state with improved validation and error handling.
        Uses a large finite number (1e6) instead of infinity for invalid states.

        Args:
            params: Input parameters tensor
        Returns:
            Normalized quantum states tensor
        """
        # Define a large but finite number to use instead of inf
        LARGE_NUM = 1e8
        SMALL_NUM = 1e-14

        try:
            # Ensure params is real-valued
            params = params.real if torch.is_complex(params) else params

            # Check for NaN or inf in input params
            if torch.any(torch.isnan(params)) or torch.any(torch.isinf(params)):
                # Replace NaN/inf with LARGE_NUM (maintaining sign for inf)
                params = params.clone()
                inf_mask = torch.isinf(params)
                nan_mask = torch.isnan(params)

                inf_count = torch.sum(inf_mask).item()
                nan_count = torch.sum(nan_mask).item()
                if inf_count > 0 or nan_count > 0:
                    logger.warning(f"Found {inf_count} inf and {nan_count} NaN values in params, replacing them")

                # Replace inf with signed LARGE_NUM
                params[inf_mask] = torch.sign(params[inf_mask]) * LARGE_NUM
                # Replace NaN with small random numbers
                params[nan_mask] = torch.rand_like(params[nan_mask]) * SMALL_NUM

            # Split into real and imaginary parts
            real_parts = params[:, : self.N]
            imag_parts = params[:, self.N :]

            # Create complex states using real-valued tensors
            states = torch.complex(real_parts.float(), imag_parts.float())

            # Calculate norms with extra validation
            abs_squared = torch.abs(states) ** 2
            # Replace any invalid values in abs_squared
            abs_squared = torch.where(
                torch.isnan(abs_squared),
                torch.ones_like(abs_squared) * SMALL_NUM,
                abs_squared,
            )
            abs_squared = torch.where(
                torch.isinf(abs_squared),
                torch.ones_like(abs_squared) * LARGE_NUM,
                abs_squared,
            )

            norms = torch.sqrt(torch.sum(abs_squared, dim=1, keepdim=True))

            # Handle zero or invalid norms
            norms = torch.where(
                (norms == 0) | torch.isnan(norms),
                torch.ones_like(norms) * SMALL_NUM,
                norms,
            )
            norms = torch.where(
                torch.isinf(norms),
                torch.ones_like(norms) * torch.sqrt(torch.tensor(LARGE_NUM)),
                norms,
            )

            # Normalize states
            states = states / norms

            # Final validation check
            if torch.any(torch.isnan(states)) or torch.any(torch.isinf(states)):
                invalid_mask = torch.any(
                    torch.isnan(states) | torch.isinf(states), dim=1, keepdim=True
                )
                invalid_count = torch.sum(invalid_mask).item()
                logger.warning(f"Found {invalid_count} invalid states after normalization, using fallback states")
                
                # Create a valid fallback state that will give a large but finite value
                # when used in calculations
                fallback_states = torch.complex(
                    torch.ones_like(states.real) * torch.sqrt(LARGE_NUM / self.N),
                    torch.zeros_like(states.imag),
                )
                states = torch.where(invalid_mask, fallback_states, states)

            return states

        except Exception as e:
            error_msg = f"Error in params_to_state_gpu: {str(e)}"
            logger.error(error_msg)
            # Return a fallback state with large magnitude in case of any unexpected errors
            batch_size = params.shape[0]
            fallback = torch.complex(
                torch.ones((batch_size, self.N), device=params.device)
                * torch.sqrt(LARGE_NUM / self.N),
                torch.zeros((batch_size, self.N), device=params.device),
            )
            return fallback


class OptimizationCallback(Callback):
    """
    Callback to record optimization progress during NSGA-II.
    """

    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []
        self.logger = setup_logger()
        self.logger.info("Optimization callback initialized")

    def notify(self, algorithm):
        """
        Record current objective function values.

        Args:
            algorithm: The current state of the optimization algorithm.
        """
        generation = algorithm.n_gen
        if hasattr(algorithm, 'opt') and algorithm.opt is not None:
            opt_F = algorithm.opt.get("F")
            if opt_F is not None:
                self.data["F"].append(opt_F.copy())
                min_f1 = opt_F[:, 0].min() if len(opt_F) > 0 else float('inf')
                min_f2 = opt_F[:, 1].min() if len(opt_F) > 0 else float('inf')
                self.logger.info(f"Generation {generation}: Min SQE={min_f1:.6f}, Min Fidelity={min_f2:.6f}, " +
                              f"Population size={len(opt_F)}")
            else:
                self.logger.warning(f"Generation {generation}: No optimal solutions found yet")
        else:
            self.logger.warning(f"Generation {generation}: Algorithm optimization data not available")


def optimize_quantum_state_gpu_cpu(
    N,
    u,
    c,
    k,
    projector=None,
    initial_kets=None,
    pop_size=500,
    max_generations=2000,
    num_workers=None,
    verbose=True,
    tolerance=5e-4,
    parity="even",
    device_id=0,
):
    """
    Perform a parallel GPU-CPU hybrid optimization of quantum states using NSGA-II.

    Args:
        N (int): Dimension of the Fock space.
        u (float): Parameter for quantum operations.
        c (float): Parameter for quantum operations.
        k (int): Parameter for quantum operations.
        projector (Qobj, optional): QuTiP projector. Defaults to None.
        initial_kets (list, optional): List of initial QuTiP ket states. Defaults to None.
        pop_size (int, optional): Population size for NSGA-II. Defaults to 500.
        max_generations (int, optional): Maximum number of generations. Defaults to 2000.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        verbose (bool, optional): If True, print detailed progress. Defaults to True.
        tolerance (float, optional): Tolerance for termination criteria. Defaults to 5e-4.
        parity (str, optional): Parity of the quantum state, either "even" or "odd". Defaults to "even".
        device_id (int, optional): GPU device ID to use. Defaults to 0.

    Returns:
        The optimization result from pymoo's minimize function.
    """
    logger = setup_logger()
    
    # Validate device_id
    if not torch.cuda.is_available():
        error_msg = "No CUDA-capable GPU devices found"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    if device_id >= torch.cuda.device_count():
        error_msg = f"GPU device {device_id} not found. Available devices: 0-{torch.cuda.device_count()-1}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Set the GPU device
    torch.cuda.set_device(device_id)

    if verbose:
        logger.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")

    # If no initial kets are provided, generate a set of candidates.
    if initial_kets is None:
        logger.info("No initial kets provided, generating candidates")
        initial_kets = [
            operator_new(N, u, 0, c, k).groundstate()[1],
            operator_new(N, u, np.pi, c, k).groundstate()[1],
        ]
        cats = [
            (
                (displace(N, u / np.sqrt(2)) + c * displace(N, -u / np.sqrt(2)))
                * squeeze(N, 1)
                * basis(N, 0)
            ).unit()
            for c in range(-1000, 1001)
        ]
        initial_kets.extend(cats)
        logger.info(f"Generated {len(initial_kets)} initial candidate states")

    # Validate and convert the initial states.
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

    # Instantiate the optimization problem.
    logger.info("Creating optimization problem instance")
    problem = GPUQuantumParetoProblem(N, u, c, k, parity, projector, device_id)

    class MultiInitialStateSampling(FloatRandomSampling):
        """
        Custom sampling that uses provided initial states combined with random sampling.
        """

        def _do(self, problem, n_samples, **kwargs):
            if n_samples <= len(initial_states):
                logger.debug(f"Using {n_samples} initial states for sampling")
                return np.vstack(initial_states[:n_samples])
            logger.debug(f"Using {len(initial_states)} initial states and {n_samples - len(initial_states)} random samples")
            random_samples = super()._do(
                problem, n_samples - len(initial_states), **kwargs
            )
            initial_states_array = np.vstack(initial_states)
            return np.vstack([initial_states_array, random_samples])

    if num_workers is None:
        num_workers = mp.cpu_count() // 2
    logger.info(f"Using {num_workers} CPU workers for parallel evaluation")

    # Create a multiprocessing pool.
    pool = mp.Pool(processes=num_workers)

    def parallel_evaluation(X, return_values_of="auto", **kwargs):
        """
        Evaluate objective functions in parallel by splitting candidate solutions into chunks.

        Args:
            X (np.ndarray): Candidate solutions.

        Returns:
            dict: Dictionary with key "F" containing evaluated objective values.
        """
        out = {"F": np.zeros((X.shape[0], 2))}
        chunk_size = max(1, X.shape[0] // num_workers)
        chunks = [X[i : i + chunk_size] for i in range(0, X.shape[0], chunk_size)]
        logger.debug(f"Splitting {X.shape[0]} candidates into {len(chunks)} chunks for parallel evaluation")
        results = []
        for i, chunk in enumerate(chunks):
            out_chunk = {"F": None}
            problem._evaluate(chunk, out_chunk)
            results.append(out_chunk["F"])
            logger.debug(f"Chunk {i+1}/{len(chunks)} evaluated")
        out["F"] = np.vstack(results)
        return out

    # Overwrite the problem's evaluation method.
    problem.evaluate = parallel_evaluation

    callback = OptimizationCallback()

    # Instantiate our custom termination object.
    termination = DefaultMultiObjectiveTermination(
        xtol=tolerance,
        cvtol=tolerance,
        ftol=tolerance,
        period=50,
        n_max_gen=max_generations,
        n_max_evals=max_generations * pop_size,
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

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MultiInitialStateSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.25, eta=12),
        eliminate_duplicates=True,
    )

    logger.info("Starting optimization")
    res = minimize(
        problem, algorithm, termination, seed=1, callback=callback, verbose=verbose
    )
    logger.info("Optimization completed")

    pool.close()
    pool.join()
    logger.info("Worker pool closed and joined")

    if parity == "even":
        out_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}"
    elif parity == "odd":
        out_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}_odd"
    
    logger.info(f"Creating output directory: {out_dir}")
    from src.helpers import create_optimization_animation

    animation_path = f"{out_dir}/pareto_front"
    logger.info(f"Creating optimization animation at {animation_path}")
    create_optimization_animation(
        callback.data["F"],
        animation_path,
    )

    return res


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Starting main program")
    
    if torch.cuda.is_available():
        device_id = 0
        logger.info(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        error_msg = "No CUDA-capable GPU devices found"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        logger.info("Starting optimization with default parameters")
        res = optimize_quantum_state_gpu_cpu(
            N=20,
            u=3,
            c=10,
            k=100,
            pop_size=500,
            max_generations=5000,
            verbose=True,
            device_id=device_id,
        )
        logger.info("Optimization completed successfully")
    except Exception as e:
        logger.error(f"Error in main program: {str(e)}", exc_info=True)
        raise