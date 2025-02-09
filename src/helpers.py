from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
from qutip import wigner, basis
from src.qutip_quantum_ops import p0_projector, measure_mode, beam_splitter
from src.cuda_quantum_ops import GPUQuantumOps
from matplotlib import colors
from tqdm import tqdm
from src.nsga_ii import optimize_quantum_state_gpu_cpu


def create_optimization_animation(F_history, filename):
    """Create animation of the optimization process"""
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        F = F_history[frame]
        ax.scatter(F[:, 0], F[:, 1], c="blue", alpha=0.5)
        ax.set_xlabel("SQE Squeezing")
        ax.set_ylabel("Output Fidelity")
        ax.set_title(f"Pareto Front Evolution (Generation {frame + 1})")

    anim = FuncAnimation(fig, update, frames=len(F_history), repeat=False)
    anim.save(filename + ".mp4", writer="ffmpeg")
    plt.close()

    # Interpolate the final Pareto front with a smoother spline
    F = F_history[-1]
    sorted_indices = np.argsort(F[:, 0])
    sorted_F = F[sorted_indices]
    f_boundary = interp1d(
        sorted_F[:, 0],
        sorted_F[:, 1],
        kind="linear",  # Use cubic interpolation for smoother curve
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Create vector and PNG plots of the final Pareto front
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(F[:, 0], F[:, 1], c="blue", alpha=0.6, s=10, label="Pareto Points")  # Smaller points
    x_new = np.linspace(np.min(F[:, 0]), np.max(F[:, 0]), 500)
    y_new = f_boundary(x_new)
    ax.plot(x_new, y_new, c="red", label="Interpolated Boundary")
    ax.set_xlabel("SQE Squeezing")
    ax.set_ylabel("Output Fidelity")
    ax.set_title("Pareto Front (Final Generation)")
    ax.legend()
    fig.savefig(filename + "_final_generation.svg")
    fig.savefig(filename + "_final_generation.png")
    plt.close()


def analyze_result(result):
    """Comprehensive analysis of the optimization result with focus on the boundary function."""

    # Get the Pareto front solutions (F contains objective values)
    F = result.F

    # Sort points by sqe squeezing (first objective) to get boundary function
    sorted_indices = np.argsort(F[:, 0])
    sorted_F = F[sorted_indices]
    sorted_X = result.X[sorted_indices]  # Decision variables (parameters)

    # Create interpolation function for the boundary
    from scipy.interpolate import interp1d

    f_boundary = interp1d(
        sorted_F[:, 0],
        sorted_F[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=(np.nan, np.nan),
    )

    # Extract key metrics
    metrics = {
        # Objective space information
        "min_sqe_squeezing": np.min(F[:, 0]),
        "max_sqe_squeezing": np.max(F[:, 0]),
        "min_fidelity": np.min(F[:, 1]),
        "max_fidelity": np.max(F[:, 1]),
        # Number of Pareto optimal solutions
        "n_solutions": len(F),
        # Range of the Pareto front
        "sqe_squeezing_range": np.max(F[:, 0]) - np.min(F[:, 0]),
        "fidelity_range": np.max(F[:, 1]) - np.min(F[:, 1]),
        # Boundary function
        "boundary_function": f_boundary,
        "boundary_points": sorted_F,
        # Original result object attributes
        "pop": result.pop,  # Final population
        "X": result.X,  # Decision variables of optimal solutions
        "F": result.F,  # Objective values of optimal solutions
        "history": result.history,  # History of the optimization
    }

    return metrics


def get_boundary_value(metrics, sqe_squeezing):
    """Get the boundary (minimum fidelity) value for a given sqe squeezing."""
    return metrics["boundary_function"](sqe_squeezing)


def analyze_optimization_result(result):
    """Complete analysis of optimization results."""
    # Get comprehensive analysis
    metrics = analyze_result(result)

    # Print key findings
    print("\nPareto Front Analysis:")
    print(f"Number of Pareto optimal solutions: {metrics['n_solutions']}")
    print(f"\nObjective Ranges:")
    print(
        f"Sqe Squeezing: [{metrics['min_sqe_squeezing']:.4f}, {metrics['max_sqe_squeezing']:.4f}]"
    )
    print(
        f"Output Fidelity: [{metrics['min_fidelity']:.4f}, {metrics['max_fidelity']:.4f}]"
    )

    # Example of using the boundary function
    test_point = (metrics["min_sqe_squeezing"] + metrics["max_sqe_squeezing"]) / 2
    bound = get_boundary_value(metrics, test_point)
    print(f"\nBoundary value at sqe squeezing = {test_point:.4f}: {bound:.4f}")

    return metrics


def save_metrics(metrics, filename):
    """
    Save metrics to a JSON file, handling non-serializable components.

    Args:
        metrics (dict): Metrics dictionary from analyze_result
        filename (str): Output filename (will append .json if not present)
    """
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    # Create serializable version of metrics
    serializable_metrics = {
        "min_sqe_squeezing": float(metrics["min_sqe_squeezing"]),
        "max_sqe_squeezing": float(metrics["max_sqe_squeezing"]),
        "min_fidelity": float(metrics["min_fidelity"]),
        "max_fidelity": float(metrics["max_fidelity"]),
        "n_solutions": int(metrics["n_solutions"]),
        "sqe_squeezing_range": float(metrics["sqe_squeezing_range"]),
        "fidelity_range": float(metrics["fidelity_range"]),
        # Save boundary points for reconstructing the function
        "boundary_points": metrics["boundary_points"].tolist(),
        # Save decision variables and objective values
        "X": metrics["X"].tolist(),
        "F": metrics["F"].tolist(),
    }

    # Save to file
    with open(filename, "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    print(f"Metrics saved to {filename}")


def load_metrics(filename):
    """
    Load metrics from a JSON file and reconstruct non-serializable components.

    Args:
        filename (str): Input filename (will append .json if not present)

    Returns:
        dict: Complete metrics dictionary with reconstructed components
    """
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    # Load from file
    with open(filename, "r") as f:
        loaded_metrics = json.load(f)

    # Convert lists back to numpy arrays
    loaded_metrics["boundary_points"] = np.array(loaded_metrics["boundary_points"])
    loaded_metrics["X"] = np.array(loaded_metrics["X"])
    loaded_metrics["F"] = np.array(loaded_metrics["F"])

    # Reconstruct boundary function
    sorted_F = loaded_metrics["boundary_points"]
    loaded_metrics["boundary_function"] = interp1d(
        sorted_F[:, 0],
        sorted_F[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=(np.nan, np.nan),
    )

    return loaded_metrics


# Function to verify saved metrics
def verify_metrics(original_metrics, loaded_metrics, test_points=None):
    """
    Verify that loaded metrics match the original ones.

    Args:
        original_metrics (dict): Original metrics before saving
        loaded_metrics (dict): Metrics after loading from file
        test_points (array-like, optional): Specific points to test boundary function
    """
    if test_points is None:
        test_points = np.linspace(
            original_metrics["min_sqe_squeezing"],
            original_metrics["max_sqe_squeezing"],
            5,
        )

    print("Verifying metrics.")
    print("\nScalar values:")
    for key in [
        "min_sqe_squeezing",
        "max_sqe_squeezing",
        "min_fidelity",
        "max_fidelity",
    ]:
        print(
            f"{key}: original = {original_metrics[key]:.6f}, "
            f"loaded = {loaded_metrics[key]:.6f}"
        )

    print("\nBoundary function values:")
    for x in test_points:
        orig_val = original_metrics["boundary_function"](x)
        loaded_val = loaded_metrics["boundary_function"](x)
        print(f"At x = {x:.4f}: original = {orig_val:.6f}, loaded = {loaded_val:.6f}")

def animate_boundary_states(N,u,c,k,file_name, save_as='animation.mp4', blend=False, title='Quantum State Animation'):
    """
    Create an animation of all the quantum states in the metrics['X'] Pareto boundary with a progress bar.
    The states are sorted in ascending order based on the expectation value of the operator_new operator.
    
    Parameters:
    metrics (dict): A dictionary containing the Pareto boundary data, including 'X' which is a list of quantum states.
    save_as (str): The filename to save the animation as, either 'animation.gif' or 'animation.mp4'.
    blend (bool): Whether to include blending between frames.
    title (str): The title of the animation.
    """
    
    projector = p0_projector(N)
    quantum_ops = GPUQuantumOps(N)
    
    metrics = load_metrics(file_name)
    # Convert the metrics['X'] list of parameters into a list of qutip states
    states = [quantum_ops.params_to_qutip(state) for state in metrics['X']]
    catfids = [tup for tup in metrics['F']]
    
    # Sort the catfids based on the first value of each tuple
    sorted_catfids = sorted(enumerate(catfids), key=lambda x: x[1][0])
    
    # Extract the sorted catfids and the indices
    sorted_catfids = [tup for _, tup in sorted_catfids]
    sort_indices = [i for i, _ in sorted(enumerate(catfids), key=lambda x: x[1][0])]
    
    # Apply the same permutation to the states list
    sorted_states = [states[i] for i in sort_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(title)
    
    xvec = np.linspace(-8, 8, 100)
    yvec = np.linspace(-8, 8, 100)
    
    cmap = plt.colormaps.get_cmap('inferno')
    norm = colors.TwoSlopeNorm(vmin=-0.23, vcenter=0, vmax=0.23)
    
    def update(frame):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        # Get current state and fidelities
        state = sorted_states[frame]
        cf = sorted_catfids[frame]
        
        # Calculate Wigner functions
        W = wigner(state, xvec, yvec)
        
        # Breeding state Wigner function
        W_breeding = wigner(measure_mode(N, beam_splitter(N, state, basis(N,0)), projector, 1), xvec, yvec)
        
        # Optional blending
        if blend and frame > 0:
            prev_state = sorted_states[frame - 1]
            prev_W = wigner(prev_state, xvec, yvec)
            W = 0.5 * W + 0.5 * prev_W
        
        # Plot first Wigner function (current state)
        cont1 = ax1.contourf(xvec, yvec, W, 30, cmap=cmap, norm=norm)
        ax1.set_title(f"Input Wigner Function\n<O> = {cf[0]:.4f}, a = {u:0.0e}, c = {c:0.0e}")
        ax1.grid(False)
        
        # Plot second Wigner function (breeding state)
        cont2 = ax2.contourf(xvec, yvec, W_breeding, 100, cmap=cmap, norm=norm)
        ax2.set_title(f"Output Wigner Function\nF = {cf[1]:.4f}")
        ax2.grid(False)
        
        return (cont1, cont2)
    
    # Use tqdm to display a progress bar
    with tqdm(total=len(sorted_states), desc="Generating boundary state animation") as pbar:
        def update_with_progress(frame):
            update(frame)
            pbar.update(1)
            return []
        
        ani = FuncAnimation(fig, update_with_progress, frames=len(sorted_states), interval=200, blit=True)
        
        if save_as.endswith('.gif'):
            ani.save(save_as, writer='pillow')
        elif save_as.endswith('.mp4'):
            ani.save(save_as, writer='ffmpeg')
        else:
            raise ValueError("save_as must be either 'animation.gif' or 'animation.mp4'")
    
    plt.close()


def run(
    gpu_id=0, N=30, u=3, c=10, k=100, pop_size=500, max_generations=2000, verbose=True, tolerance=5e-4
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    output_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(
        f"{output_dir}/metrics.json"
    ):
        print(
            f"GPU-{gpu_id}: Optimization for N = {N}, u = {u}, c = {c}, k = {k} already finished."
        )
        return None
    else:
        print(
            f"GPU-{gpu_id}: Starting optimization for N = {N}, u = {u}, c = {c}, k = {k}, pop_size = {pop_size}, max_generations = {max_generations}"
        )

        start_time = time.time()

        result = optimize_quantum_state_gpu_cpu(
            N=N,
            u=u,
            c=c,
            k=k,
            projector=None,
            pop_size=pop_size,
            max_generations=max_generations,
            verbose=verbose,
            tolerance=tolerance,
        )

        end_time = time.time()

        print(
            f"GPU-{gpu_id}: Optimization completed in {end_time - start_time:.2f} seconds"
        )

        # Analyze and save metrics
        metrics = analyze_result(result)
        metrics["time"] = end_time - start_time

        save_metrics(metrics, f"{output_dir}/metrics")
        print(f"GPU-{gpu_id}: Metrics successfully to {output_dir}/metrics.json")
        print(f"GPU-{gpu_id}: Generating animation for Pareto boundary states.")
        animate_boundary_states(N,u,c,k,f"{output_dir}/metrics.json", save_as=f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}/boundary_states.mp4", title=f"Boundary States for N = {N}, u = {u}, c = {c}, k = {k}")
        print(f"GPU-{gpu_id}: Animation saved to {output_dir}/metrics_N{N}_u{u}_c{c}_k{k}.mp4")