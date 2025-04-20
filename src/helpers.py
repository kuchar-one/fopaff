from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from qutip import wigner, basis
from src.qutip_quantum_ops import p0_projector, measure_mode, beam_splitter
from src.cuda_quantum_ops import GPUQuantumOps
from matplotlib import colors
from tqdm import tqdm
from src.nsga_ii import optimize_quantum_state_gpu_cpu
from src.logger import setup_logger

# Set up logger
logger = setup_logger()


def create_optimization_animation(F_history, filename):
    """
    Create an animation and static plots of the optimization process.

    This function generates both an animated visualization of the Pareto front
    evolution and static plots of the final generation. The animation shows how
    the Pareto front develops over generations, while the static plots provide
    detailed views of the final optimization state.

    Args:
        F_history (list): List of numpy arrays containing objective values for
            each generation. Each array has shape (n_solutions, 2) where the
            columns are [SQE_squeezing, output_fidelity].
        filename (str): Base filename for saving the outputs. Will be used to
            create multiple files with different extensions.

    Note:
        Creates the following files:
        - {filename}.mp4: Animation of Pareto front evolution
        - {filename}_final_generation.svg: Vector plot of final Pareto front
        - {filename}_final_generation.png: Bitmap plot of final Pareto front
        
        The final generation plots include both scatter points and an
        interpolated boundary line for better visualization.
    """
    logger.info(f"Creating optimization animation at {filename}")
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
    logger.info(f"Saved animation to {filename}.mp4")

    # Interpolate the final Pareto front with a smoother spline
    F = F_history[-1]
    sorted_indices = np.argsort(F[:, 0])
    sorted_F = F[sorted_indices]
    f_boundary = interp1d(
        sorted_F[:, 0],
        sorted_F[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Create vector and PNG plots of the final Pareto front
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(F[:, 0], F[:, 1], c="blue", alpha=0.6, s=10, label="Pareto Points")
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
    logger.info(f"Saved final generation plots to {filename}_final_generation.svg/png")


def analyze_result(result):
    """
    Perform comprehensive analysis of optimization results with focus on Pareto front.

    This function analyzes the optimization results by extracting key metrics,
    creating boundary functions, and organizing the data for further analysis.
    It focuses on both the objective space and decision variable space.

    Args:
        result: Optimization result object containing:
            - F: Objective values for Pareto optimal solutions
            - X: Decision variables for Pareto optimal solutions
            - pop: Final population
            - history: Optimization history

    Returns:
        dict: Dictionary containing analysis metrics:
            - min/max_sqe_squeezing: Range of SQE squeezing values
            - min/max_fidelity: Range of output fidelity values
            - n_solutions: Number of Pareto optimal solutions
            - sqe_squeezing_range: Total range of squeezing values
            - fidelity_range: Total range of fidelity values
            - boundary_function: Interpolation function for Pareto front
            - boundary_points: Actual points on Pareto front
            - Original result attributes (pop, X, F, history)

    Note:
        The boundary_function is created using linear interpolation and can
        be used to estimate the minimum fidelity achievable for any given
        squeezing value within the range.
    """
    logger.info("Analyzing optimization results")

    # Get the Pareto front solutions (F contains objective values)
    F = result.F

    # Sort points by sqe squeezing (first objective) to get boundary function
    sorted_indices = np.argsort(F[:, 0])
    sorted_F = F[sorted_indices]
    sorted_X = result.X[sorted_indices]  # Decision variables (parameters)

    # Create interpolation function for the boundary
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

    logger.debug(f"Analyzed metrics: min_sqe_squeezing={metrics['min_sqe_squeezing']:.4f}, "
                f"max_sqe_squeezing={metrics['max_sqe_squeezing']:.4f}, "
                f"min_fidelity={metrics['min_fidelity']:.4f}, "
                f"max_fidelity={metrics['max_fidelity']:.4f}")

    return metrics


def get_boundary_value(metrics, sqe_squeezing):
    """
    Get the minimum achievable fidelity for a given squeezing value.

    This function uses the interpolated Pareto front boundary to estimate
    the minimum fidelity achievable for a specific squeezing value.

    Args:
        metrics (dict): Metrics dictionary from analyze_result containing
            the boundary_function.
        sqe_squeezing (float): The squeezing value to evaluate.

    Returns:
        float: The estimated minimum fidelity achievable at the given
            squeezing value. Returns NaN if the squeezing value is
            outside the interpolation range.

    Note:
        This function uses linear interpolation between known points on
        the Pareto front. Values outside the range of known solutions
        will return NaN.
    """
    logger.debug(f"Getting boundary value for sqe_squeezing={sqe_squeezing}")
    return metrics["boundary_function"](sqe_squeezing)


def analyze_optimization_result(result):
    """
    Perform complete analysis of optimization results and log key findings.

    This function serves as a high-level interface for analyzing optimization
    results. It combines the detailed analysis with logging of key metrics
    and example boundary value calculations.

    Args:
        result: Optimization result object containing Pareto front solutions
            and optimization history.

    Returns:
        dict: Complete metrics dictionary from analyze_result, containing
            all analysis results and boundary functions.

    Note:
        - Logs comprehensive information about the optimization results
        - Includes example boundary value calculation at midpoint
        - Useful for quick assessment of optimization performance
    """
    logger.info("Performing full optimization result analysis")
    
    # Get comprehensive analysis
    metrics = analyze_result(result)

    # Log key findings
    logger.info(f"Number of Pareto optimal solutions: {metrics['n_solutions']}")
    logger.info(f"Sqe Squeezing range: [{metrics['min_sqe_squeezing']:.4f}, {metrics['max_sqe_squeezing']:.4f}]")
    logger.info(f"Output Fidelity range: [{metrics['min_fidelity']:.4f}, {metrics['max_fidelity']:.4f}]")

    # Example of using the boundary function
    test_point = (metrics['min_sqe_squeezing'] + metrics['max_sqe_squeezing']) / 2
    bound = get_boundary_value(metrics, test_point)
    logger.info(f"Boundary value at sqe squeezing = {test_point:.4f}: {bound:.4f}")

    return metrics


def save_metrics(metrics, filename):
    """
    Save optimization metrics to a JSON file with proper serialization.

    This function handles the serialization of optimization metrics to JSON
    format, properly converting non-serializable components (like numpy arrays
    and functions) to serializable formats.

    Args:
        metrics (dict): Metrics dictionary from analyze_result containing
            both serializable and non-serializable components.
        filename (str): Output filename. Will append .json if not present.

    Note:
        - Automatically adds .json extension if not present
        - Converts numpy arrays to lists for serialization
        - Omits non-serializable components (like functions)
        - Creates a clean, human-readable JSON file with proper indentation
        - Logs success or failure of the save operation
    """
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    logger.info(f"Saving metrics to {filename}")

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
    try:
        with open(filename, "w") as f:
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Successfully saved metrics to {filename}")
    except Exception as e:
        logger.error(f"Error saving metrics to {filename}: {str(e)}")


def load_metrics(filename):
    """
    Load optimization metrics from a JSON file and reconstruct components.

    This function loads previously saved metrics from a JSON file and
    reconstructs non-serializable components like numpy arrays and
    interpolation functions.

    Args:
        filename (str): Input filename. Will append .json if not present.

    Returns:
        dict: Complete metrics dictionary with reconstructed components,
            including numpy arrays and interpolation functions.

    Raises:
        Exception: If there are errors during file loading or reconstruction.

    Note:
        - Automatically adds .json extension if not present
        - Converts lists back to numpy arrays
        - Reconstructs the boundary interpolation function
        - Preserves the original data structure of the metrics
    """
    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    logger.info(f"Loading metrics from {filename}")

    try:
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

        logger.info(f"Successfully loaded metrics from {filename}")
        return loaded_metrics
    except Exception as e:
        logger.error(f"Error loading metrics from {filename}: {str(e)}")
        raise


def verify_metrics(original_metrics, loaded_metrics, test_points=None):
    """
    Verify consistency between original and loaded metrics.

    This function compares the original metrics with those loaded from a file
    to ensure data integrity and proper reconstruction of components.

    Args:
        original_metrics (dict): Original metrics before saving.
        loaded_metrics (dict): Metrics after loading from file.
        test_points (array-like, optional): Specific points at which to test
            the boundary function. If None, uses 5 evenly spaced points across
            the squeezing range.

    Note:
        - Compares scalar values directly
        - Tests boundary function at multiple points
        - Verifies array shapes and contents
        - Useful for ensuring data persistence integrity
    """
    logger.info("Verifying metrics consistency between original and loaded data")
    
    if test_points is None:
        test_points = np.linspace(
            original_metrics["min_sqe_squeezing"],
            original_metrics["max_sqe_squeezing"],
            5,
        )

    # Log scalar values comparison
    for key in [
        "min_sqe_squeezing",
        "max_sqe_squeezing",
        "min_fidelity",
        "max_fidelity",
        "n_solutions",
        "sqe_squeezing_range",
        "fidelity_range",
    ]:
        logger.debug(f"Comparing {key}: original={original_metrics[key]}, loaded={loaded_metrics[key]}")


class PlateauTwoSlopeNorm(colors.TwoSlopeNorm):
    """
    Custom normalization for color maps with a plateau region.

    This class extends matplotlib's TwoSlopeNorm to create a color mapping
    that includes a plateau region around the center value. This is useful
    for visualizing quantum state data where values near zero should be
    treated similarly.

    Args:
        vcenter (float): Center value for the normalization.
        plateau_size (float): Size of the plateau region around vcenter.
        vmin (float, optional): Minimum value for normalization.
        vmax (float, optional): Maximum value for normalization.

    Note:
        - Creates a "dead zone" around vcenter where values map to the same color
        - Useful for Wigner function visualization
        - Helps highlight significant deviations from the center value
    """
    def __init__(self, vcenter, plateau_size, vmin=None, vmax=None):
        super().__init__(vcenter=vcenter, vmin=vmin, vmax=vmax)
        self.plateau_size = plateau_size

    def __call__(self, value, clip=None):
        """
        Transform values to the [0, 1] interval for colormapping.

        Args:
            value: Input values to normalize.
            clip (bool, optional): Whether to clip values outside [vmin, vmax].

        Returns:
            float or array: Normalized values in [0, 1].
        """
        # Get the basic normalization
        result = super().__call__(value, clip=clip)

        # Apply plateau modification
        mask = np.abs(value - self.vcenter) < self.plateau_size
        if isinstance(result, np.ndarray):
            result[mask] = 0.5
        elif mask:
            result = 0.5
        return result

    def inverse(self, value):
        """
        Inverse transform from [0, 1] to original scale.

        Args:
            value: Normalized values in [0, 1].

        Returns:
            float or array: Values in original scale.
        """
        result = super().inverse(value)
        # Note: This is an approximation as the plateau creates a many-to-one mapping
        return result


def animate_boundary_states(N, u, c, k, file_name, save_as='animation.mp4', blend=False, title='Quantum State Animation'):
    """
    Create an animation of quantum states along the Pareto boundary.

    This function generates an animation showing how quantum states evolve
    along the Pareto optimal boundary, visualizing their Wigner functions
    and other properties.

    Args:
        N (int): Dimension of the Fock space.
        u (float): Displacement amplitude parameter.
        c (float): Coupling strength parameter.
        k (int): Order parameter.
        file_name (str): Base filename for loading metrics.
        save_as (str, optional): Output animation filename. Defaults to 'animation.mp4'.
        blend (bool, optional): Whether to blend between states. Defaults to False.
        title (str, optional): Title for the animation. Defaults to 'Quantum State Animation'.

    Note:
        - Creates a visualization of quantum state evolution
        - Uses Wigner function representation
        - Can show smooth transitions between states if blend=True
        - Saves the animation in MP4 format
        - Shows progress bar during generation
    """
    logger.info(f"Creating boundary states animation from {file_name}, saving to {save_as}")
    logger.debug(f"Animation parameters: N={N}, u={u}, c={c}, k={k}, blend={blend}")
    
    projector = p0_projector(N)
    quantum_ops = GPUQuantumOps(N)
    
    try:
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

        logger.info(f"Processing {len(sorted_states)} quantum states for animation")

        fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1, 0.05]})
        plt.suptitle(title)
        
        xvec = np.linspace(-8, 8, 100)
        yvec = np.linspace(-8, 8, 100)
        
        cmap = plt.colormaps.get_cmap('inferno')
        
        class PlateauTwoSlopeNorm(colors.TwoSlopeNorm):
            def __init__(self, vcenter, plateau_size, vmin=None, vmax=None):
                """
                A modified TwoSlopeNorm that maintains a constant color within 
                a specified range around vcenter before transitioning to the endpoints.
                
                Parameters
                ----------
                vcenter : float
                    The central value that defines the plateau midpoint
                plateau_size : float
                    The total width of the plateau region (vcenter ± plateau_size/2)
                vmin : float, optional
                    The minimum value in the normalization
                vmax : float, optional
                    The maximum value in the normalization
                """
                super().__init__(vcenter=vcenter, vmin=vmin, vmax=vmax)
                self.plateau_size = plateau_size
                
            def __call__(self, value, clip=None):
                """
                Map values to the interval [0, 1], maintaining a constant value
                within the plateau region.
                """
                result, is_scalar = self.process_value(value)
                self.autoscale_None(result)
                
                if not self.vmin <= self.vcenter <= self.vmax:
                    raise ValueError("vmin, vcenter, vmax must increase monotonically")
                    
                # Define plateau boundaries
                plateau_lower = self.vcenter - self.plateau_size/2
                plateau_upper = self.vcenter + self.plateau_size/2
                
                # Create interpolation points including plateau region
                x_points = [self.vmin, plateau_lower, plateau_upper, self.vmax]
                y_points = [0, 0.5, 0.5, 1]
                
                result = np.ma.masked_array(
                    np.interp(result, x_points, y_points, left=-np.inf, right=np.inf),
                    mask=np.ma.getmask(result))
                    
                if is_scalar:
                    result = np.atleast_1d(result)[0]
                return result
                
            def inverse(self, value):
                if not self.scaled():
                    raise ValueError("Not invertible until both vmin and vmax are set")
                    
                plateau_lower = self.vcenter - self.plateau_size/2
                plateau_upper = self.vcenter + self.plateau_size/2
                
                x_points = [0, 0.5, 0.5, 1]
                y_points = [self.vmin, plateau_lower, plateau_upper, self.vmax]
                
                return np.interp(value, x_points, y_points, left=-np.inf, right=np.inf)
            
            
        norm = PlateauTwoSlopeNorm(vcenter=0, plateau_size=0.03, vmin=-0.23, vmax=0.23)
        
        # Create a dummy contour plot just for the colorbar
        dummy_data = wigner(measure_mode(N, beam_splitter(N, sorted_states[0], basis(N, 0)), projector, 1), xvec, yvec)
        dummy_cont = ax1.contourf(xvec, yvec, dummy_data, 30, cmap=cmap, norm=norm)
        plt.colorbar(dummy_cont, cax=cax, orientation='vertical')
        ax1.clear()  # Clear the dummy plot
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            state = sorted_states[frame]
            cf = sorted_catfids[frame]
            
            W = wigner(state, xvec, yvec)
            W_breeding = wigner(measure_mode(N, beam_splitter(N, state, basis(N, 0)), projector, 1), xvec, yvec)
            
            if blend and frame > 0:
                prev_state = sorted_states[frame - 1]
                prev_W = wigner(prev_state, xvec, yvec)
                W = 0.5 * W + 0.5 * prev_W
            
            # Use same normalization for both plots
            cont1 = ax1.contourf(xvec, yvec, W, 30, cmap=cmap, norm=norm)
            ax1.set_title(f"Input Wigner Function\n<O> = {cf[0]:.4f}")
            ax1.grid(False)
            
            cont2 = ax2.contourf(xvec, yvec, W_breeding, 100, cmap=cmap, norm=norm)
            ax2.set_title(f"Output Wigner Function\nF = {cf[1]:.4f}")
            ax2.grid(False)
            
            return (cont1, cont2)
        
        with tqdm(total=len(sorted_states), desc="Generating boundary state animation") as pbar:
            def update_with_progress(frame):
                result = update(frame)
                pbar.update(1)
                return result
            
            ani = FuncAnimation(fig, update_with_progress, frames=len(sorted_states), interval=200, blit=True)
            
            if save_as.endswith('.gif'):
                ani.save(save_as, writer='pillow')
            elif save_as.endswith('.mp4'):
                ani.save(save_as, writer='ffmpeg')
            else:
                logger.error(f"Invalid save_as format: {save_as}. Must be .gif or .mp4")
                raise ValueError("save_as must be either 'animation.gif' or 'animation.mp4'")
        
        plt.close()
        logger.info(f"Successfully created and saved animation to {save_as}")
    except Exception as e:
        logger.error(f"Error creating boundary states animation: {str(e)}")
        raise


def run(
    gpu_id=0, N=30, u=3, c=10, k=100, pop_size=500, max_generations=2000, verbose=True, tolerance=5e-4, parity="even"
):
    """
    Run the quantum state optimization with proper GPU device management.
    
    Args:
        gpu_id (int): GPU device ID to use
        N (int): Dimension of the Hilbert space
        u (float): Parameter for quantum operations
        c (float): Parameter for quantum operations
        k (int): Parameter for quantum operations
        pop_size (int): Population size for NSGA-II
        max_generations (int): Maximum number of generations
        verbose (bool): If True, print detailed progress
        tolerance (float): Tolerance for termination criteria
        parity (str): Parity of the quantum state, either "even" or "odd"
    
    Returns:
        None or optimization result
    """
    import os
    import torch
    
    logger.info(f"Starting quantum state optimization run with parameters: N={N}, u={u}, c={c}, k={k}")
    logger.info(f"Optimization settings: pop_size={pop_size}, max_generations={max_generations}, tolerance={tolerance}, parity={parity}")
    
    # Validate GPU availability
    if not torch.cuda.is_available():
        logger.error("No CUDA-capable GPU devices found")
        raise RuntimeError("No CUDA-capable GPU devices found")
    
    # Check if requested GPU is available
    device_count = torch.cuda.device_count()
    if gpu_id >= device_count:
        logger.error(f"GPU device {gpu_id} not found. Available devices: 0-{device_count-1}")
        raise ValueError(f"GPU device {gpu_id} not found. Available devices: 0-{device_count-1}")
    
    # Set CUDA device - both environment variable and PyTorch setting
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    
    #logger.info(f"Using GPU device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Prepare output directory
    if parity == "even":
        output_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}"
    elif parity == "odd":
        output_dir = f"output/{max_generations}_maxgens_{pop_size}_individuals_N{N}_u{u}_c{c}_k{k}_odd"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Check if optimization has already been completed
    if os.path.isfile(f"{output_dir}/metrics.json"):
        logger.info(f"Optimization for N = {N}, u = {u}, c = {c}, k = {k} already finished. Skipping.")
        return None
    else:
        logger.info(f"Starting optimization for N = {N}, u = {u}, c = {c}, k = {k}, pop_size = {pop_size}, max_generations = {max_generations}")
        
        # Log CUDA memory information
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        logger.info(f"GPU-{gpu_id} Memory: Total {total_memory:.2f} GB, Allocated {allocated_memory:.2f} GB, Reserved {reserved_memory:.2f} GB")
        
        start_time = time.time()
        
        try:
            # Run the optimization with explicit device_id parameter
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
                parity=parity,
                device_id=gpu_id  # Pass the explicit device_id
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            
            # Analyze and save metrics
            metrics = analyze_result(result)
            metrics["time"] = elapsed_time
            metrics["device_id"] = gpu_id
            metrics["device_name"] = torch.cuda.get_device_name(gpu_id)
            save_metrics(metrics, f"{output_dir}/metrics")
            
            logger.info(f"Metrics successfully saved to {output_dir}/metrics.json")
            
            # Generate visualization
            logger.info(f"Generating animation for Pareto boundary states")
            animate_boundary_states(
                N, u, c, k, 
                f"{output_dir}/metrics.json", 
                save_as=f"{output_dir}/boundary_states.mp4", 
                title=f"Boundary States for N = {N}, u = {u}, c = {c}, k = {k}"
            )
            
            logger.info(f"Animation saved to {output_dir}/boundary_states.mp4")
            
            # Clean up any remaining CUDA memory
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
            
            return result
        except Exception as e:
            logger.error(f"Error during optimization run: {str(e)}", exc_info=True)
            raise