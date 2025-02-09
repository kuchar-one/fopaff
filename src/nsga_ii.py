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
from src.qutip_quantum_ops import operator_new, p0_projector
from qutip import expect, displace, squeeze, basis, Qobj


class GPUQuantumParetoProblem(Problem):
    """
    Multi-objective optimization problem for quantum state optimization using GPU.

    Attributes:
        N (int): Dimension of the Hilbert space.
        u (float): Parameter for quantum operations.
        c (float): Parameter for quantum operations.
        k (int): Parameter for quantum operations.
        quantum_ops (GPUQuantumOps): GPU-accelerated quantum operations handler.
        squeezing_bound (float): Bound used to decide if a candidate is valid.
        operator: Precomputed operator used in evaluation.
        projector (torch.Tensor): Projector tensor used for fidelity calculations.
    """

    def __init__(self, N, u, c, k, parity, projector=None):
        self.N = N
        self.u = u
        self.c = c
        self.k = k
        self.quantum_ops = GPUQuantumOps(N)
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
            raise ValueError(f"Parity must be either 'even' or 'odd', not '{parity}'")

        if projector is None:
            self.projector = torch.tensor(
                p0_projector(self.N).full(), dtype=torch.complex64
            ).cuda()
        elif isinstance(projector, Qobj):
            self.projector = torch.tensor(
                projector.full(), dtype=torch.complex64
            ).cuda()
        else:
            raise ValueError("Projector must be a QuTiP Qobj")

        n_var = 2 * N  # Real and imaginary parts
        bound = 10

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
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - allocated_memory

        element_size = 4 * self.N * 2  # Approximate size in bytes for complex64 elements
        safe_batch_size = free_memory // element_size

        return max(1, safe_batch_size)  # Ensure at least a batch size of 1

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objective functions for a given set of candidate solutions.

        Args:
            x (np.ndarray): Array of candidate solutions.
            out (dict): Dictionary to store evaluation results with key "F".
        """
        try:
            x_gpu = torch.tensor(x, dtype=torch.complex64).cuda()
            f = torch.zeros((x_gpu.shape[0], 2), device="cuda")

            optimal_batch_size = self._check_memory()

            for i in range(0, len(x_gpu), optimal_batch_size):
                batch = x_gpu[i : i + optimal_batch_size]
                states = self.params_to_state_gpu(batch)

                for j, state in enumerate(states):
                    try:
                        cat_sq, out_fid = self.quantum_ops.catfid_gpu(
                            state, self.u, self.parity, self.c, self.k, self.projector, self.operator
                        )

                        # Ensure valid numbers.
                        if torch.isnan(cat_sq) or torch.isinf(cat_sq):
                            cat_sq = torch.tensor(float("inf"), device="cuda")
                        if torch.isnan(out_fid) or torch.isinf(out_fid):
                            out_fid = torch.tensor(1.0, device="cuda")

                        # If candidate exceeds squeezing bound, assign worst values.
                        if np.real(cat_sq.item()) > self.squeezing_bound:
                            f[i + j, 0] = torch.tensor(1e8, device="cuda")
                            f[i + j, 1] = torch.tensor(1.0, device="cuda")
                        else:
                            f[i + j, 0] = torch.abs(cat_sq)
                            f[i + j, 1] = torch.abs(out_fid)
                    except Exception as e:
                        print(f"Error in evaluation: {str(e)}")
                        f[i + j, 0] = torch.tensor(float("inf"), device="cuda")
                        f[i + j, 1] = torch.tensor(0, device="cuda")

            out["F"] = f.cpu().numpy()
        except Exception as e:
            print(f"Fatal error in evaluation: {str(e)}")
            out["F"] = np.full((x.shape[0], 2), np.inf)

    def params_to_state_gpu(self, params):
        """
        Convert parameter vectors into normalized quantum states on the GPU.

        Uses large finite numbers to replace any infinities or NaNs in the process.

        Args:
            params (torch.Tensor): Input parameter tensor.

        Returns:
            torch.Tensor: Tensor of normalized quantum states.
        """
        LARGE_NUM = 1e8
        SMALL_NUM = 1e-14

        try:
            # Ensure parameters are real-valued if complex.
            params = params.real if torch.is_complex(params) else params

            # Replace any NaN or inf values.
            if torch.any(torch.isnan(params)) or torch.any(torch.isinf(params)):
                params = params.clone()
                inf_mask = torch.isinf(params)
                nan_mask = torch.isnan(params)
                params[inf_mask] = torch.sign(params[inf_mask]) * LARGE_NUM
                params[nan_mask] = torch.rand_like(params[nan_mask]) * SMALL_NUM

            # Split into real and imaginary parts.
            real_parts = params[:, : self.N]
            imag_parts = params[:, self.N :]
            states = torch.complex(real_parts.float(), imag_parts.float())

            # Compute and validate norms.
            abs_squared = torch.abs(states) ** 2
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

            # Normalize states.
            states = states / norms

            # Final check: replace invalid states if needed.
            if torch.any(torch.isnan(states)) or torch.any(torch.isinf(states)):
                invalid_mask = torch.any(
                    torch.isnan(states) | torch.isinf(states), dim=1, keepdim=True
                )
                fallback_states = torch.complex(
                    torch.ones_like(states.real) * torch.sqrt(LARGE_NUM / self.N),
                    torch.zeros_like(states.imag),
                )
                states = torch.where(invalid_mask, fallback_states, states)

            return states

        except Exception as e:
            print(f"Error in params_to_state_gpu: {str(e)}")
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

    def notify(self, algorithm):
        """
        Record current objective function values.
        
        Args:
            algorithm: The current state of the optimization algorithm.
        """
        self.data["F"].append(algorithm.opt.get("F").copy())

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
):
    """
    Perform a parallel GPU-CPU hybrid optimization of quantum states using NSGA-II.
    
    Args:
        N (int): Dimension of the Hilbert space.
        u (float): Parameter for quantum operations.
        c (float): Parameter for quantum operations.
        k (int): Parameter for quantum operations.
        projector (Qobj, optional): QuTiP projector. Defaults to None.
        initial_kets (list, optional): List of initial QuTiP ket states. Defaults to None.
        pop_size (int, optional): Population size for NSGA-II. Defaults to 500.
        max_generations (int, optional): Maximum number of generations. Defaults to 2000.
        num_workers (int, optional): Number of CPU workers. Defaults to None.
        verbose (bool, optional): If True, print detailed progress. Defaults to True.
    
    Returns:
        The optimization result from pymoo's minimize function.
    """
    # If no initial kets are provided, generate a set of candidates.
    if initial_kets is None:
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
    
    # Validate and convert the initial states.
    initial_states = []
    for initial_ket in initial_kets:
        if not isinstance(initial_ket, Qobj) or not initial_ket.isket:
            raise ValueError("All initial_kets must be valid QuTiP ket states")
        if initial_ket.dims[0][0] != N:
            raise ValueError(
                f"Initial ket dimension ({initial_ket.dims[0][0]}) does not match N ({N})"
            )
        amplitudes = initial_ket.full().flatten()
        initial_state = np.concatenate([amplitudes.real, amplitudes.imag])
        initial_states.append(initial_state)
    
    # Instantiate the optimization problem.
    problem = GPUQuantumParetoProblem(N, u, c, k, parity, projector)
    
    class MultiInitialStateSampling(FloatRandomSampling):
        """
        Custom sampling that uses provided initial states combined with random sampling.
        """
        
        def _do(self, problem, n_samples, **kwargs):
            if n_samples <= len(initial_states):
                return np.vstack(initial_states[:n_samples])
            random_samples = super()._do(problem, n_samples - len(initial_states), **kwargs)
            initial_states_array = np.vstack(initial_states)
            return np.vstack([initial_states_array, random_samples])
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
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
        results = []
        for chunk in chunks:
            out_chunk = {"F": None}
            problem._evaluate(chunk, out_chunk)
            results.append(out_chunk["F"])
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
        print("\nOptimization Settings:")
        print(f"Population size: {pop_size}")
        print(f"Maximum generations: {max_generations}")
        print(f"Number of workers: {num_workers}")
        print("\nTermination Criteria:")
        print(f"xtol: {termination.x.termination.tol}")
        print(f"cvtol: {termination.cv.termination.tol}")
        print(f"ftol: {termination.f.termination.tol}")
        print(f"n_max_gen: {termination.max_gen.n_max_gen}\n")
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MultiInitialStateSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.25, eta=12),
        eliminate_duplicates=True,
    )
    
    res = minimize(
        problem, algorithm, termination, seed=1, callback=callback, verbose=verbose
    )
    
    pool.close()
    pool.join()
    
    if parity == "even":
        out_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}"
    elif parity == "odd":
        out_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}_odd"
    from src.helpers import create_optimization_animation
    create_optimization_animation(
        callback.data["F"],
        f"{out_dir}/pareto_front",
    )
    
    return res


if __name__ == "__main__":
    res = optimize_quantum_state_gpu_cpu(N=20, u=3, c=10, k=100, pop_size=500, max_generations=5000, verbose=True)
