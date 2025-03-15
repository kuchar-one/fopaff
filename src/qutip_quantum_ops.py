import os
os.environ['OMP_NUM_THREADS'] = '6'
import matplotlib.pyplot as plt
import qutip as qt
import numpy as np
from mpmath import sqrt, pi, jtheta, exp, fac
from scipy.optimize import minimize_scalar
from scipy.special import hermite, factorial, gamma


def high_dim_generator(u, phi, c, k):
    """
    Generate and save the high dimensional operator O_2(N, a, phi, c, k) = (position(N)^2 - a^2 * qeye(N))^2 + c * a * 4^k * (fac(k))^2 / (pi * fac(2 * k)) * (-1/4)^k * (expm(-1j * (momentum(N) * a + phi/2)) - expm(1j * (momentum(N) * a + phi/2)))^(2 * k).

    Parameters
    ----------
    a : float
        A parameter of the operator.
    phi : float
        A parameter of the operator.
    c : float
        A parameter of the operator.
    k : int
        A parameter of the operator.

    Returns
    -------
    Qobj
        The operator O_2(N, a, phi, c, k).
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


def operator_new(N, u, phi, c, k):
    """
    Load or generate a high-dimensional operator and return its truncated version.

    This function attempts to load a previously saved high-dimensional operator
    from a file. If the file does not exist, it generates the operator using the
    provided parameters and saves it. The operator is then truncated to the specified
    dimension N and returned.

    Parameters
    ----------
    N : int
        The dimension to which the operator should be truncated.
    a : float
        A parameter used in the generation of the operator.
    phi : float
        A parameter used in the generation of the operator.
    c : float
        A parameter used in the generation of the operator.
    k : int
        A parameter used in the generation of the operator.

    Returns
    -------
    Qobj
        The truncated high-dimensional operator.
    """
    filename = f"cache/operators/high_dim_u{u:.2f}_phi{phi:.2f}_c{c:.2f}_k{k:.2f}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_generator(u, phi, c, k)

    return qt.Qobj(high_dim.full()[:N, :N])


def high_dim_gkp_generator():
    """
    Generate and save a high-dimensional GKP squeezing operator.

    The operator is calculated in a 5000-dimensional Fock space and then saved to a file.

    Returns
    -------
    oper
        The high-dimensional gkp squeezing operator.
    """
    dim = 5000
    sin_term1 = +1j * qt.position(dim) * np.sqrt(np.pi) / 2
    sin_term2 = +1j * qt.momentum(dim) * np.sqrt(np.pi)
    high_dim = 2 * operator_sin(sin_term1) ** 2 + 2 * operator_sin(sin_term2) ** 2

    print(f"Generating pre-truncation form of the GKP Operator...")

    qt.qsave(high_dim, f"cache/operators/high_dim_gkp")

    return high_dim


def gkp_operator_new(N):
    """
    Load or generate a high-dimensional operator and return its truncated version.

    This function attempts to load a previously saved high-dimensional operator
    from a file. If the file does not exist, it generates the operator using the
    provided parameters and saves it. The operator is then truncated to the specified
    dimension N and returned.

    Parameters
    ----------
    N : int
        The dimension to which the operator should be truncated.

    Returns
    -------
    Qobj
        The truncated high-dimensional operator.
    """
    filename = f"cache/operators/high_dim_gkp"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_gkp_generator()

    return qt.Qobj(high_dim.full()[:N, :N])


def gaussian_limit(u, c):
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


def beam_splitter(N, one_in, two_in, theta=np.pi / 4):
    """Optimized beam splitter interaction with caching

    Args:
        N (int): truncated Fock space dimension
        one_in (ket or oper): first mode input
        two_in (ket or oper): second mode input
        theta (float, optional): BS parameter, balanced default

    Returns:
        oper: two-mode output density matrix
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


def measure_mode(N, two_mode_dm, projector, mode):
    """Optimized projection measurement of two mode state

    Args:
        N (int): truncated Fock space dimension
        two_mode_dm (oper): two mode state density operator
        projector (oper): measurement projection
        mode (1 or 2): mode number

    Returns:
        oper: conditional output density matrix in untouched mode
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


def breeding(N, rounds, input_state, projector):
    """Breeding protocol for generating a GKP state.

    Args:
        N (int): truncated Fock space dimension.
        rounds (int): number of breeding rounds.
        input_state (oper): input state.
        projector (oper): measurement projector.

    Returns:
        oper: output density matrix of the protocol.
    """
    if rounds == 0:
        return input_state

    else:
        temp = beam_splitter(N, input_state, input_state)
        new = measure_mode(N, temp, projector, 1)
        output_state = breeding(N, rounds - 1, new, projector)

    return output_state


def squeezed_cat(N, a, r):
    """
    Generate a squeezed Schrödinger cat state.

    Parameters
    ----------
    N : int
        The dimension of the Fock space.
    a : float
        The displacement amplitude.
    r : float
        The squeezing parameter.

    Returns
    -------
    Qobj
        A normalized squeezed Schrödinger cat state.
    """
    plus = qt.displace(N, a / np.sqrt(2)) * qt.squeeze(N, r) * qt.basis(N, 0)
    minus = qt.displace(N, -a / np.sqrt(2)) * qt.squeeze(N, r) * qt.basis(N, 0)
    return (plus + minus) / (plus + minus).norm()


def operator_sin(sin_term):
    """
    Calculate the sine operator from a given operator.

    Parameters
    ----------
    sin_term : oper
        The input operator.

    Returns
    -------
    oper
        The sine operator.
    """
    return (sin_term.expm() - (-sin_term).expm()) / (2j)


def operator_cos(cos_term):
    """
    Calculate the cosine operator from a given operator.

    Parameters
    ----------
    cos_term : oper
        The input operator.

    Returns
    -------
    oper
        The cosine operator.
    """
    return (cos_term.expm() + (-cos_term).expm()) / 2


def find_optimal_p0(N, r=0.0):
    """
    Find the optimal squeezing for the p-eigenstate projector, given a dimension N.

    This function attempts to find the optimal squeezing value for the p-eigenstate projector
    such that the probability of the first N Fock states is closest to 0.99.

    Parameters
    ----------
    N : int
        The dimension of the projector.

    r : float, optional
        The starting value for the squeezing. Defaults to 0.

    Returns
    -------
    Qobj
        The optimal p-eigenstate projector of dimension N.
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

    qt.qsave(projector, f"cache/operators/p0_projector_N{N}")

    return projector


def p0_projector(N, r=0.0):
    """
    Load or generate the p0 projector for a given dimension N.

    This function checks for a precomputed p0 projector saved in a file. If it
    exists, it loads the projector. Otherwise, it calculates the optimal squeezing
    parameter to generate the projector and saves it for future use.

    Parameters
    ----------
    N : int
        The dimension of the Fock space.
    r : float, optional
        Initial squeezing parameter guess (default is 0.0).

    Returns
    -------
    Qobj
        The truncated p0 projector for the specified dimension N.
    """
    filename = f"cache/operators/p0_projector_N{N}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = find_optimal_p0(N, r)

    return qt.Qobj(high_dim.full()[:N, :N])


def catfid(N, state, a, c, k, projector):

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


def catgkp(N, state, rounds, c, k, projector):

    u = 2 * np.sqrt(2) * np.sqrt(np.pi) * 2 ** ((rounds - 3) / 2)
    operator = operator_new(N, u, 0, c, k)
    cat_squeezing = qt.expect(operator, state)
    gkp_state = breeding(N, rounds, state, projector)
    gkp_operator = gkp_operator_new(N)
    gkp_squeezing = qt.expect(gkp_operator, gkp_state)

    return (cat_squeezing, gkp_squeezing)


def construct_initial_state(N, desc, param):
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


def construct_operator(N, desc, param):
    if desc == "GKP":
        return gkp_operator_new(N)
    elif desc == "cat":
        return operator_new(N, param, 0, 10, 100)
    else:
        raise ValueError(f"Unknown operator description: {desc}")
