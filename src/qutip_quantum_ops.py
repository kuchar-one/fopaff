import os
os.environ['OMP_NUM_THREADS'] = '6'
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
from mpmath import sqrt, pi, jtheta, exp, fac
from scipy.optimize import minimize_scalar
from scipy.special import hermite, factorial, gamma
from typing import Tuple, Union, Optional
from qutip import Qobj


def high_dim_generator(u: float, phi: float, c: float, k: int) -> Qobj:
    """
    Generate and save a high-dimensional quantum operator for cat state analysis.

    This function constructs the operator O₂(N,u,φ,c,k) in a high-dimensional
    Fock space (2500 dimensions) and saves it to disk for later use. The operator
    is defined as:

    O₂ = (x² - u²)² + cu(4ᵏk!²/πΓ(k+½))(-¼)ᵏ(e^(-i(px+φ/2)) - e^(i(px+φ/2)))^(2k)

    where x and p are the position and momentum operators.

    Args:
        u (float): Displacement amplitude parameter.
        phi (float): Phase parameter.
        c (float): Coupling strength parameter.
        k (int): Order parameter affecting the operator's behavior.

    Returns:
        qutip.Qobj: The high-dimensional operator in QuTiP format.

    Note:
        - Uses a 2500-dimensional Fock space for high precision
        - Caches the operator to disk for future use
        - Cache directory is created if it doesn't exist
        - Filename encodes all parameters for unique identification
    """
    dim = 2500
    plus = 1j * (qt.momentum(dim) * u + phi / 2)
    minus = -1j * (qt.momentum(dim) * u + phi / 2)

    high_dim = (qt.position(dim) ** 2 - u**2 * qt.qeye(dim)) ** 2 + c * u * gamma(
        k + 1
    ) / (np.sqrt(np.pi) * gamma(k + 0.5)) * (-1 / 4) ** k * (
        minus.expm() - plus.expm()
    ) ** (
        2 * k
    )

    print(
        f"Generating pre-truncation form of the SQE Operator for u = {u}, phi = {phi}, c = {c}, k = {k}..."
    )

    os.makedirs("cache/operators", exist_ok=True)
    qt.qsave(
        high_dim, f"cache/operators/high_dim_u{u:.2f}_phi{phi:.2f}_c{c:.2f}_k{k:.2f}"
    )

    return high_dim


def operator_new(N: int, u: float, phi: float, c: float, k: int) -> Qobj:
    """
    Load or generate a truncated high-dimensional quantum operator.

    This function manages the creation and caching of quantum operators used for
    cat state analysis. It either loads a previously cached operator or generates
    a new one, then truncates it to the desired dimension.

    Args:
        N (int): Dimension to truncate the operator to.
        u (float): Displacement amplitude parameter.
        phi (float): Phase parameter.
        c (float): Coupling strength parameter.
        k (int): Order parameter affecting the operator's behavior.

    Returns:
        qutip.Qobj: The truncated operator in QuTiP format.

    Note:
        - First attempts to load from cache to avoid expensive computation
        - If not found in cache, generates using high_dim_generator
        - Truncates the operator from 2500 dimensions to N dimensions
        - Preserves operator properties while reducing computational cost
    """
    filename = f"cache/operators/high_dim_u{u:.2f}_phi{phi:.2f}_c{c:.2f}_k{k:.2f}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_generator(u, phi, c, k)

    return qt.Qobj(high_dim.full()[:N, :N])


def high_dim_gkp_generator() -> Qobj:
    """
    Generate and save a high-dimensional GKP (Gottesman-Kitaev-Preskill) squeezing operator.

    This function constructs the GKP squeezing operator in a 5000-dimensional
    Fock space. The operator is defined as:

    O_GKP = 2sin²(√π x/2) + 2sin²(√π p)

    where x and p are the position and momentum operators.

    Returns:
        qutip.Qobj: The high-dimensional GKP operator.

    Note:
        - Uses a 5000-dimensional Fock space for high precision
        - Caches the operator to disk for future use
        - The operator measures how well a state approximates a GKP state
        - Higher expectation values indicate better GKP state approximation
    """
    dim = 5000
    sin_term1 = +1j * qt.position(dim) * np.sqrt(np.pi) / 2
    sin_term2 = +1j * qt.momentum(dim) * np.sqrt(np.pi)
    high_dim = 2 * operator_sin(sin_term1) ** 2 + 2 * operator_sin(sin_term2) ** 2

    print(f"Generating pre-truncation form of the GKP Operator...")

    qt.qsave(high_dim, f"cache/operators/high_dim_gkp")

    return high_dim


def gkp_operator_new(N: int) -> Qobj:
    """
    Load or generate a truncated GKP squeezing operator.

    This function manages the creation and caching of the GKP squeezing operator.
    It either loads a previously cached operator or generates a new one, then
    truncates it to the desired dimension.

    Args:
        N (int): Dimension to truncate the operator to.

    Returns:
        qutip.Qobj: The truncated GKP operator in QuTiP format.

    Note:
        - First attempts to load from cache to avoid expensive computation
        - If not found in cache, generates using high_dim_gkp_generator
        - Truncates the operator from 5000 dimensions to N dimensions
        - Used to measure how well states approximate GKP states
    """
    filename = f"cache/operators/high_dim_gkp"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_gkp_generator()

    return qt.Qobj(high_dim.full()[:N, :N])


def gaussian_limit(u: float, c: float) -> float:
    """
    Calculate the Gaussian limit for cat state squeezing.

    This function computes the theoretical limit of squeezing achievable with
    Gaussian operations for a given set of parameters. It uses numerical
    optimization to find the minimum value.

    Args:
        u (float): Displacement amplitude parameter.
        c (float): Coupling strength parameter.

    Returns:
        float: The Gaussian limit value.

    Note:
        - Uses scipy's minimize_scalar for optimization
        - Considers both analytical and numerical bounds
        - Important for benchmarking non-Gaussian advantages
        - Returns the minimum of two different approaches
    """
    return min(
        [
            u * c / np.pi,
            minimize_scalar(
                lambda r: float(
                    u**4
                    + 3 / (4 * exp(4 * r))
                    - u**2 / exp(2 * r)
                    + u * c / pi * jtheta(3, -1 / 2 * pi, exp(-(u**2 * exp(2 * r))))
                ),
                bounds=[-10, 10],
                method="bounded",
            ).fun,
        ]
    )


def beam_splitter(
    N: int,
    one_in: Qobj,
    two_in: Qobj,
    theta: float = np.pi / 4
) -> Qobj:
    """
    Simulate a beam splitter interaction between two quantum states.

    This function implements an optimized beam splitter transformation with
    caching for efficiency. The beam splitter is a fundamental component in
    many quantum optical protocols.

    Args:
        N (int): Truncated Fock space dimension.
        one_in (qutip.Qobj): First mode input (ket or density matrix).
        two_in (qutip.Qobj): Second mode input (ket or density matrix).
        theta (float, optional): Beam splitter mixing angle in radians.
            Defaults to π/4 (50:50 beam splitter).

    Returns:
        qutip.Qobj: Two-mode output density matrix.

    Note:
        - Automatically converts input states to density matrices
        - Caches the beam splitter unitary for efficiency
        - Uses the transformation:
          a₁ → a₁cos(θ) - a₂sin(θ)
          a₂ → a₁sin(θ) + a₂cos(θ)
        - Creates cache directory if it doesn't exist
    """
    # Convert inputs to density matrices if needed
    one_dm = qt.ket2dm(one_in) if one_in.type == "ket" else one_in
    two_dm = qt.ket2dm(two_in) if two_in.type == "ket" else two_in

    # Cache filename
    filename = f"cache/operators/beam_splitter_qutip_N{N}_theta{theta:.2f}"

    # Check if cached unitary exists
    if os.path.isfile(f"{filename}.qu"):
        unitary = qt.qload(filename)
    else:
        # Create destruction operators using QutIP's tensor
        destroy_one = qt.tensor(qt.destroy(N), qt.identity(N))
        destroy_two = qt.tensor(qt.identity(N), qt.destroy(N))

        # Compute the unitary operator
        generator = -theta * (
            destroy_one.dag() * destroy_two - destroy_two.dag() * destroy_one
        )
        unitary = generator.expm()

        # Save the unitary to cache
        os.makedirs("cache/operators", exist_ok=True)
        qt.qsave(unitary, filename)

    # Compute final state
    input_dm = qt.tensor(one_dm, two_dm)
    return unitary.dag() * input_dm * unitary


def measure_mode(N: int, two_mode_dm: Qobj, projector: Qobj, mode: int) -> Qobj:
    """
    Perform an optimized projective measurement on a two-mode quantum state.

    This function implements a projective measurement on one mode of a two-mode
    quantum state, followed by partial trace over the measured mode. The result
    is the post-measurement state of the unmeasured mode.

    Args:
        N (int): Truncated Fock space dimension.
        two_mode_dm (qutip.Qobj): Two-mode quantum state density operator.
        projector (qutip.Qobj): Measurement projection operator.
        mode (int): Which mode to measure (1 or 2).

    Returns:
        qutip.Qobj: Normalized post-measurement state in the unmeasured mode.

    Note:
        - Mode numbering starts at 1
        - Automatically normalizes the output state
        - Uses tensor product structure for efficient computation
        - Preserves quantum state properties
    """
    id_N = qt.identity(N)

    # Pre-compute measurement operator
    measurement = (
        qt.tensor(projector, id_N) if mode == 1 else qt.tensor(id_N, projector)
    )

    # Perform measurement and partial trace
    measured_state = two_mode_dm * measurement
    traced_state = measured_state.ptrace(1 if mode == 1 else 0)

    return traced_state.unit()


def breeding(N: int, rounds: int, input_state: Qobj, projector: Qobj) -> Qobj:
    """
    Implement the breeding protocol for generating GKP states.

    This function implements an iterative protocol that uses beam splitters and
    measurements to generate grid states. The protocol is repeated for a specified
    number of rounds to improve the quality of the output state.

    Args:
        N (int): Truncated Fock space dimension.
        rounds (int): Number of breeding protocol iterations.
        input_state (qutip.Qobj): Input quantum state.
        projector (qutip.Qobj): Measurement projector.

    Returns:
        qutip.Qobj: Output state after the breeding protocol.

    Note:
        - Uses recursive implementation for multiple rounds
        - Each round consists of:
          1. 50:50 beam splitter on two copies of the input state
          2. Measurement of one mode
          3. Using the post-measurement state for the next round
        - Returns input state unchanged if rounds = 0
    """
    if rounds == 0:
        return input_state

    else:
        temp = beam_splitter(N, input_state, input_state)
        new = measure_mode(N, temp, projector, 1)
        output_state = breeding(N, rounds - 1, new, projector)

    return output_state


def squeezed_cat(N: int, a: float, r: float) -> Qobj:
    """
    Generate a normalized squeezed Schrödinger cat state.

    This function creates a superposition of two coherent states, displaced in
    opposite directions and squeezed. The state is of the form:
    |ψ⟩ ∝ (D(a/√2) + D(-a/√2))S(r)|0⟩

    Args:
        N (int): Dimension of the Fock space.
        a (float): Displacement amplitude (total separation is 2a).
        r (float): Squeezing parameter (r > 0 for squeezing).

    Returns:
        qutip.Qobj: Normalized squeezed cat state.

    Note:
        - Uses vacuum state as the initial state
        - Applies squeezing before displacement
        - Automatically normalizes the output state
        - The state is an even cat state (+ superposition)
    """
    plus = qt.displace(N, a / np.sqrt(2)) * qt.squeeze(N, r) * qt.basis(N, 0)
    minus = qt.displace(N, -a / np.sqrt(2)) * qt.squeeze(N, r) * qt.basis(N, 0)
    return (plus + minus) / (plus + minus).norm()


def operator_sin(sin_term: Qobj) -> Qobj:
    """
    Calculate the sine of a quantum operator.

    This function implements the sine operation on a quantum operator using
    the exponential representation:
    sin(A) = (e^(iA) - e^(-iA))/(2i)

    Args:
        sin_term (qutip.Qobj): The operator to take the sine of.

    Returns:
        qutip.Qobj: The sine of the input operator.

    Note:
        - Uses matrix exponential for computation
        - Preserves operator properties
        - Useful for constructing GKP operators
    """
    return (sin_term.expm() - (-sin_term).expm()) / (2j)


def operator_cos(cos_term: Qobj) -> Qobj:
    """
    Calculate the cosine of a quantum operator.

    This function implements the cosine operation on a quantum operator using
    the exponential representation:
    cos(A) = (e^(iA) + e^(-iA))/2

    Args:
        cos_term (qutip.Qobj): The operator to take the cosine of.

    Returns:
        qutip.Qobj: The cosine of the input operator.

    Note:
        - Uses matrix exponential for computation
        - Preserves operator properties
        - Useful for constructing GKP operators
    """
    return (cos_term.expm() + (-cos_term).expm()) / 2


def find_optimal_p0(N: int, r: float = 0.0) -> Qobj:
    """
    Find the optimal squeezing for the p-eigenstate projector.

    This function iteratively searches for the optimal squeezing parameter that
    ensures the first N Fock states capture most of the p=0 eigenstate. It uses
    a larger Hilbert space (10N) for accurate computation before truncation.

    Args:
        N (int): Target dimension of the projector.
        r (float, optional): Initial squeezing parameter guess. Defaults to 0.0.

    Returns:
        qutip.Qobj: The optimal p-eigenstate projector of dimension N.

    Note:
        - Uses a 10N-dimensional space for accurate computation
        - Targets 99% probability in the first N Fock states
        - Caches the result for future use
        - Creates cache directory if it doesn't exist
        - Prints progress information during optimization
    """
    well_approximated = True
    threshold_probability = 0.99

    # Pre-compute the state for 10*N once
    basis_10N_0 = qt.basis(10 * N, 0)

    print(f"Constructing optimal p = 0 projection in N = {N} dimensions...")

    while well_approximated:
        test_state = qt.squeeze(10 * N, r) * basis_10N_0
        coefs = test_state.full().flatten()
        probability = np.sum(coefs[:N] ** 2)

        if probability < threshold_probability:
            well_approximated = False
            optimal_r = r + 0.01
            print(f"optimal p eigenket squeezing for N = {N} is {optimal_r}")
        else:
            r -= 0.01

    # Compute the final squeezed state for N
    projector = qt.ket2dm(qt.squeeze(N, optimal_r) * qt.basis(N, 0))

    os.makedirs("cache/operators", exist_ok=True)
    qt.qsave(projector, f"cache/operators/p0_projector_N{N}")

    return projector


def p0_projector(N: int, r: float = 0.0) -> Qobj:
    """
    Load or generate the p=0 eigenstate projector.

    This function manages the creation and caching of the p=0 eigenstate projector.
    It either loads a previously computed projector or generates a new one with
    optimal squeezing parameters.

    Args:
        N (int): Dimension of the Fock space.
        r (float, optional): Initial squeezing parameter guess. Defaults to 0.0.

    Returns:
        qutip.Qobj: The p=0 eigenstate projector in N dimensions.

    Note:
        - First attempts to load from cache to avoid expensive computation
        - If not found in cache, finds optimal squeezing using find_optimal_p0
        - Truncates high-dimensional results to N dimensions
        - Important for GKP state generation and verification
    """
    filename = f"cache/operators/p0_projector_N{N}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = find_optimal_p0(N, r)

    return qt.Qobj(high_dim.full()[:N, :N])


def catfid(
    N: int,
    state: Qobj,
    a: float,
    c: float,
    k: int,
    projector: Qobj
) -> Tuple[complex, float]:
    """
    Calculate cat state squeezing and output fidelity metrics.

    This function computes two key metrics for evaluating cat states:
    1. Cat squeezing: Expectation value of the cat squeezing operator
    2. Output fidelity: Overlap with an ideal squeezed cat state after measurement

    Args:
        N (int): Dimension of the Fock space.
        state (qutip.Qobj): Input quantum state to evaluate.
        a (float): Displacement amplitude parameter.
        c (float): Coupling strength parameter.
        k (int): Order parameter for the squeezing operator.
        projector (qutip.Qobj): Measurement projector.

    Returns:
        Tuple[complex, float]: (cat_squeezing, output_fidelity)
            - cat_squeezing: Expectation value of the squeezing operator
            - output_fidelity: Overlap with ideal squeezed cat state

    Note:
        - Constructs ideal state as (D(a/2) + D(-a/2))S(log(2)/2)|0⟩
        - Uses beam splitter and measurement for output state
        - Automatically normalizes states where needed
    """
    operator = operator_new(N, a, 0, c, k)
    cat_squeezing = qt.expect(operator, state)
    ideal_state = (
        (qt.displace(N, a / 2) + qt.displace(N, -a / 2))
        * qt.squeeze(N, np.log(2) / 2)
        * qt.basis(N, 0)
    ).unit()
    output_state = measure_mode(
        N, beam_splitter(N, state, qt.basis(N, 0)), projector, 1
    )
    output_fidelity = qt.fidelity(output_state, ideal_state)

    return (cat_squeezing, output_fidelity)


def catgkp(
    N: int,
    state: Qobj,
    rounds: int,
    c: float,
    k: int,
    projector: Qobj
) -> Tuple[complex, complex]:
    """
    Calculate cat and GKP squeezing metrics for a quantum state.

    This function evaluates how well a state approximates both a cat state and
    a GKP state by computing their respective squeezing parameters. The GKP
    state is generated through the breeding protocol.

    Args:
        N (int): Dimension of the Fock space.
        state (qutip.Qobj): Input quantum state to evaluate.
        rounds (int): Number of breeding protocol iterations.
        c (float): Coupling strength parameter.
        k (int): Order parameter for the squeezing operator.
        projector (qutip.Qobj): Measurement projector.

    Returns:
        Tuple[complex, complex]: (cat_squeezing, gkp_squeezing)
            - cat_squeezing: Expectation value of cat squeezing operator
            - gkp_squeezing: Expectation value of GKP squeezing operator

    Note:
        - Displacement amplitude u scales with breeding rounds
        - Uses breeding protocol to generate GKP state
        - Both metrics are expectation values of respective operators
        - Higher values indicate better approximation to target states
    """
    u = 2 * np.sqrt(2) * np.sqrt(np.pi) * 2 ** ((rounds - 3) / 2)
    operator = operator_new(N, u, 0, c, k)
    cat_squeezing = qt.expect(operator, state)
    gkp_state = breeding(N, rounds, state, projector)
    gkp_operator = gkp_operator_new(N)
    gkp_squeezing = qt.expect(gkp_operator, gkp_state)

    return (cat_squeezing, gkp_squeezing)


def construct_initial_state(N: int, desc: str, param: Union[float, int]) -> Qobj:
    """
    Construct an initial quantum state based on a description and parameter.

    This function creates various types of quantum states commonly used in
    quantum optics experiments and simulations. It supports Fock states,
    coherent states, squeezed states, displaced states, and cat states.

    Args:
        N (int): Dimension of the Fock space.
        desc (str): Description of the state type. Options:
            - "vacuum": Ground state |0⟩
            - "coherent": Coherent state |α⟩
            - "squeezed": Squeezed vacuum S(r)|0⟩
            - "displaced": Displaced vacuum D(α)|0⟩
            - "cat": Cat state ∝ (D(α/2) + D(-α/2))|0⟩
            - "squeezed_cat": Ground state of cat squeezing operator
            - Or a number n for Fock state |n⟩
        param (Union[float, int]): Parameter for state construction
            (e.g., α for coherent state, r for squeezing).

    Returns:
        qutip.Qobj: The constructed quantum state.

    Raises:
        ValueError: If the state description is not recognized.

    Note:
        - All states are automatically normalized
        - For Fock states, returns a weight suggestion of 0.1^n
        - Prints debug information for coherent states
        - Uses operator_new for squeezed cat states
    """
    try:
        n = int(desc)
        return qt.basis(N, n), 0.1**n
    except:
        if desc == "vacuum":
            return qt.basis(N, 0)
        elif desc == "coherent":
            print(f"Coherent state with alpha = {param}, type: {type(param)}")
            return qt.coherent(N, param)
        elif desc == "squeezed":
            return qt.squeeze(N, param) * qt.basis(N, 0)
        elif desc == "displaced":
            return qt.displace(N, param) * qt.basis(N, 0)
        elif desc == "cat":
            return (
                (qt.displace(N, param / 2) + qt.displace(N, -param / 2))
                * qt.basis(N, 0)
            ).unit()
        elif desc == "squeezed_cat":
            return operator_new(N, param, 0, 10, 100).groundstate()[1]
        else:
            raise ValueError(f"Unknown initial state description: {desc}")


def construct_operator(N: int, desc: str, param: float) -> Qobj:
    """
    Construct a quantum operator based on a description and parameter.

    This function creates various quantum operators used for state analysis
    and measurement. Currently supports GKP and cat squeezing operators.

    Args:
        N (int): Dimension of the Fock space.
        desc (str): Description of the operator type. Options:
            - "GKP": GKP squeezing operator
            - "cat": Cat squeezing operator
        param (float): Parameter for operator construction
            (e.g., displacement amplitude for cat operator).

    Returns:
        qutip.Qobj: The constructed quantum operator.

    Raises:
        ValueError: If the operator description is not recognized.

    Note:
        - Uses cached operators when available
        - For cat operators, uses default values c=10, k=100
        - GKP operator is parameter-independent
    """
    if desc == "GKP":
        return gkp_operator_new(N)
    elif desc == "cat":
        return operator_new(N, param, 0, 10, 100)
    else:
        raise ValueError(f"Unknown operator description: {desc}")
