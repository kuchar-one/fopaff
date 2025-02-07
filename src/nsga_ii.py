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
    def __init__(self, N, u, c, k, projector=None):
        self.N = N
        self.u = u
        self.c = c
        self.k = k
        self.quantum_ops = GPUQuantumOps(N)
        self.squeezing_bound = np.real(
            expect(
                operator_new(self.N, self.u, 0.0, self.c, self.k),
                operator_new(self.N, self.u, np.pi, self.c, self.k).groundstate()[1],
            )
        )
        self.operator = self.quantum_ops.operator_new_gpu(self.u, 0, self.c, self.k)

        if projector == None:
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
        #print("Coefficient bound:", bound)

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=-bound,
            xu=bound,
        )

    def _check_memory(self):
        """Check available GPU memory and adjust batch size accordingly."""
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - allocated_memory

        element_size = 4 * self.N * 2  # Approx size in bytes for complex64
        safe_batch_size = free_memory // element_size

        return max(1, safe_batch_size)  # Ensure at least batch size of 1

    def _evaluate(self, x, out, *args, **kwargs):
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
                            state, self.u, self.c, self.k, self.projector, self.operator
                        )

                        # Add validity checks
                        if torch.isnan(cat_sq) or torch.isinf(cat_sq):
                            cat_sq = torch.tensor(float("inf"), device="cuda")
                        if torch.isnan(out_fid) or torch.isinf(out_fid):
                            out_fid = torch.tensor(1.0, device="cuda")

                        # Skip points with cat_sq > yk by giving them worst possible values
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
                # Create a valid fallback state that will give a large but finite value
                # when used in calculations
                fallback_states = torch.complex(
                    torch.ones_like(states.real) * torch.sqrt(LARGE_NUM / self.N),
                    torch.zeros_like(states.imag),
                )
                states = torch.where(invalid_mask, fallback_states, states)

            return states

        except Exception as e:
            print(f"Error in params_to_state_gpu: {str(e)}")
            # Return a fallback state with large magnitude in case of any unexpected errors
            batch_size = params.shape[0]
            fallback = torch.complex(
                torch.ones((batch_size, self.N), device=params.device)
                * torch.sqrt(LARGE_NUM / self.N),
                torch.zeros((batch_size, self.N), device=params.device),
            )
            return fallback


class OptimizationCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []

    def notify(self, algorithm):
        self.data["F"].append(algorithm.opt.get("F").copy())




def optimize_quantum_state_gpu_cpu(
    N, u, c, k, projector=None, initial_kets=None, pop_size=500, max_generations=2000, num_workers=None, verbose=True
):
    """
    Parallel GPU-CPU hybrid optimization of quantum states
    """
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

    # print(f"Initial kets: {initial_kets[0:4]}")

    # Validate and convert initial states
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

    # print(f"Initial states: {initial_states[0:4]}")

    problem = GPUQuantumParetoProblem(N, u, c, k, projector)

    class MultiInitialStateSampling(FloatRandomSampling):
        def _do(self, problem, n_samples, **kwargs):
            if n_samples <= len(initial_states):
                return np.vstack(initial_states[:n_samples])
            random_samples = super()._do(
                problem, n_samples - len(initial_states), **kwargs
            )
            initial_states_array = np.vstack(initial_states)
            return np.vstack([initial_states_array, random_samples])

    if num_workers is None:
        num_workers = mp.cpu_count()

    pool = mp.Pool(processes=num_workers)

    # Create a wrapper for the evaluation function that handles pymoo's requirements
    def parallel_evaluation(X, return_values_of="auto", **kwargs):
        out = {"F": np.zeros((X.shape[0], 2))}
        chunk_size = max(1, X.shape[0] // num_workers)
        chunks = [X[i : i + chunk_size] for i in range(0, X.shape[0], chunk_size)]

        # Process chunks in parallel
        results = []
        for chunk in chunks:
            out_chunk = {"F": None}
            problem._evaluate(chunk, out_chunk)
            results.append(out_chunk["F"])

        # Combine results
        out["F"] = np.vstack(results)
        return out

    # Assign the parallel evaluation function to the problem
    problem.evaluate = parallel_evaluation

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MultiInitialStateSampling(),
        crossover=SBX(prob=0.85, eta=20),
        mutation=PM(prob=0.20, eta=15),
        eliminate_duplicates=True,
    )

    callback = OptimizationCallback()

    termination = DefaultMultiObjectiveTermination(
        n_max_gen=max_generations,  # Maximum number of generations
        xtol=0.0005,  # Tolerance for design space convergence
        ftol=0.005,   # Tolerance for objective space convergence
        period=50     # Check convergence every 50 generations
    )
    
    res = minimize(
        problem, algorithm, termination, seed=1, callback=callback, verbose=verbose
    )
    
    pool.close()
    pool.join()

    from src.helpers import create_optimization_animation
    create_optimization_animation(
        callback.data["F"],
        f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}/parreto_front",
    )

    return res