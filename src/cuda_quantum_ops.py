import sys
import os
import math
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from torch import Tensor
from qutip import (
    Qobj, destroy, identity, basis, tensor,
    fidelity, ket2dm, wigner
)

from src.cuda_helpers import DeviceMonitor
from src.qutip_quantum_ops import (
    catfid,
    operator_new,
    gkp_operator_new,
    p0_projector,
    catgkp,
)
from src.logger import setup_logger

# Set OpenMP threads for CPU operations
os.environ['OMP_NUM_THREADS'] = '6'

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = setup_logger()


class GPUQuantumOps:
    """
    GPU-accelerated quantum operations for state manipulation and measurement.
    
    This class provides GPU-accelerated implementations of common quantum operations
    including state preparation, measurements, and transformations. It uses PyTorch
    for GPU acceleration and provides compatibility with QuTiP for verification.
    
    Attributes:
        N (int): Dimension of the Hilbert space.
        monitor (DeviceMonitor): Monitor for CUDA device usage and memory.
        device (str): CUDA device identifier (e.g., 'cuda:0').
        d (torch.Tensor): Destruction operator matrix.
        identity (torch.Tensor): Identity operator matrix.
    """

    def __init__(self, N: int) -> None:
        """
        Initialize GPU quantum operations with specified Hilbert space dimension.

        Args:
            N (int): Dimension of the Hilbert space.
            
        Note:
            The class initializes on the default device (cuda:0) and pre-computes
            common operators. The actual device can be changed later using the
            GPUDeviceWrapper.
        """
        self.N = N
        self.monitor = DeviceMonitor()
        self.device = "cuda:0"  # Default device, will be updated by wrapper

        # Print initial CUDA information
        self.monitor.print_cuda_info()

        # Pre-compute operators on default device (will be moved by wrapper)
        self.d = self._create_destroy_operator().to(self.device)
        self.identity = torch.eye(N, dtype=torch.complex64).to(self.device)

    def _ensure_same_device(self, *tensors: Tensor) -> torch.device:
        """
        Ensure all provided tensors are on the same device.

        Args:
            *tensors: Variable length argument list of tensors to check.

        Returns:
            torch.device: The common device of the tensors, or the default device if no tensors provided.

        Raises:
            RuntimeError: If tensors are found on different devices.
        """
        devices = {t.device for t in tensors if torch.is_tensor(t)}
        if len(devices) > 1:
            raise RuntimeError(f"Tensors found on different devices: {devices}")
        return list(devices)[0] if devices else self.device

    def beam_splitter(
        self,
        one_in: Tensor,
        two_in: Tensor,
        theta: float = np.pi / 4
    ) -> Tensor:
        """
        Simulate a beam splitter operation on two input quantum states.

        This function applies a beam splitter transformation to two input states using
        a specified mixing angle. The transformation is implemented using a unitary
        operator that is cached for efficiency.

        Args:
            one_in (torch.Tensor): First input quantum state (density matrix or state vector).
            two_in (torch.Tensor): Second input quantum state (density matrix or state vector).
            theta (float, optional): Beam splitter mixing angle in radians. Defaults to π/4.

        Returns:
            torch.Tensor: Resulting quantum state after beam splitter transformation.
            
        Note:
            - The beam splitter unitary is cached to disk for reuse.
            - If matrix exponential fails, falls back to Taylor series approximation.
            - Input states are automatically converted to density matrices if given as vectors.
        """
        # Cache handling with device awareness
        cache_dir = os.path.join("cache", "operators")
        os.makedirs(cache_dir, exist_ok=True)
        filename = f"beam_splitter_unitary_theta{theta:.4f}_N{self.N}.pt"
        cache_path = os.path.join(cache_dir, filename)

        if os.path.exists(cache_path):
            U = torch.load(cache_path, weights_only=False).to(self.device)
        else:
            one_in = one_in.to(self.device)
            two_in = two_in.to(self.device)

            # Create beam splitter operator
            d1 = self.tensor_product(self.d, self.identity)
            d2 = self.tensor_product(self.identity, self.d)
            op = -theta * (torch.matmul(d1.T.conj(), d2) - torch.matmul(d2.T.conj(), d1))

            try:
                U = torch.matrix_exp(op)
                torch.save(U.cpu(), cache_path)
            except Exception as e:
                logger.warning(f"Matrix exponential failed: {str(e)}, using Taylor series approximation")
                U = torch.eye(op.shape[0], dtype=op.dtype, device=self.device)
                op_power = torch.eye(op.shape[0], dtype=op.dtype, device=self.device)
                for i in range(1, 10):
                    op_power = torch.matmul(op_power, op) / i
                    U = U + op_power

        # Convert state vectors to density matrices if needed
        if len(one_in.shape) == 1:
            one_in = torch.outer(one_in, one_in.conj())
        if len(two_in.shape) == 1:
            two_in = torch.outer(two_in, two_in.conj())

        # Apply beam splitter transformation
        input_dm = self.tensor_product(one_in, two_in)
        result = torch.matmul(torch.matmul(U.T.conj(), input_dm), U)

        return result.to(self.device)

    def breeding_gpu(
        self,
        rounds: int,
        input_state: Tensor,
        projector: Tensor
    ) -> Tensor:
        """
        Simulate the breeding protocol to generate a GKP state.

        This function implements an iterative breeding protocol that applies beam splitter
        operations and measurements to generate grid states. The protocol is repeated
        for a specified number of rounds.

        Args:
            rounds (int): Number of breeding protocol iterations.
            input_state (torch.Tensor): Input quantum state (density matrix or state vector).
            projector (torch.Tensor): Measurement projector matrix.

        Returns:
            torch.Tensor: Final quantum state after breeding protocol.
            
        Note:
            The breeding protocol consists of:
            1. Applying a 50:50 beam splitter to two copies of the input state
            2. Measuring one mode using the provided projector
            3. Using the post-measurement state as input for the next round
        """
        input_state = input_state.to(self.device)
        projector = projector.to(self.device)

        device = self._ensure_same_device(input_state, projector)
        input_state = input_state.to(device)
        projector = projector.to(device)

        if rounds == 0:
            return input_state

        temp = self.beam_splitter(input_state, input_state)
        new_state = self.measure_mode(temp, projector, 1)
        output_state = self.breeding_gpu(rounds - 1, new_state, projector)

        return output_state.to(self.device)

    def measure_mode(
        self,
        two_mode_dm: Tensor,
        projector: Tensor,
        mode: int
    ) -> Tensor:
        """
        Perform a projective measurement on one mode of a two-mode quantum state.

        Args:
            two_mode_dm (torch.Tensor): Two-mode quantum state density matrix.
            projector (torch.Tensor): Projector for the measurement.
            mode (int): Which mode to measure (1 or 2).

        Returns:
            torch.Tensor: Post-measurement quantum state.
            
        Raises:
            ValueError: If mode is not 1 or 2.
            
        Note:
            The measurement operator is cached to disk for efficiency.
            The measurement projects onto the subspace defined by the projector
            and traces out the measured mode.
        """
        two_mode_dm = two_mode_dm.to(self.device)
        projector = projector.to(self.device)

        # Cache handling for measurement operator
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"measurement_mode{mode}_N{self.N}.pt")

        if os.path.exists(cache_path):
            measurement = torch.load(cache_path, weights_only=False).to(self.device)
        else:
            if mode == 1:
                measurement = torch.kron(projector, self.identity)
            elif mode == 2:
                measurement = torch.kron(self.identity, projector)
            else:
                raise ValueError("Mode must be 1 or 2.")
            torch.save(measurement.cpu(), cache_path)

        measured_dm = torch.matmul(two_mode_dm, measurement)
        measured_dm = torch.matmul(measured_dm, measurement.conj().T)

        if mode == 1:
            conditional_dm = self.partial_trace(measured_dm, mode=0)
        elif mode == 2:
            conditional_dm = self.partial_trace(measured_dm, mode=1)

        trace_val = torch.trace(conditional_dm)
        if trace_val != 0:
            conditional_dm = conditional_dm / trace_val

        return conditional_dm.to(self.device)

    def get_free_memory(self) -> float:
        """
        Get the amount of free memory in the current PyTorch device.

        Returns:
            float: The amount of free memory in megabytes (MB).
            
        Note:
            Returns 0 if CUDA is not available.
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)  # Convert bytes to MB
        return 0

    def _create_destroy_operator(self, offset: int = 0) -> Tensor:
        """
        Create the destruction (lowering) operator in the Fock basis.

        This operator acts on quantum states by lowering their energy level,
        transforming |n⟩ to √n|n-1⟩. It is a fundamental operator in quantum optics.

        Args:
            offset (int, optional): The lowest number state included in the finite
                number state representation. Defaults to 0.

        Returns:
            torch.Tensor: The lowering operator as a complex matrix.
            
        Note:
            The matrix elements are given by:
            ⟨n-1|a|n⟩ = √(n + offset)
            where n ranges from 1 to N-1.
        """
        # Initialize the destruction operator matrix with zeros on the GPU
        d = torch.zeros((self.N, self.N), dtype=torch.complex64).cuda()

        # Populate the off-diagonal elements with the square roots of integers
        for n in range(1, self.N):
            d[n - 1, n] = torch.sqrt(torch.tensor(n + offset, dtype=torch.float32)).to(
                d.device
            )

        return d

    def expect(self, operator: Tensor, state: Tensor) -> Tensor:
        """
        Calculate the expectation value of an operator with respect to a state.

        Computes either ⟨ψ|A|ψ⟩ for pure states or Tr(ρA) for mixed states.

        Args:
            operator (torch.Tensor): Complex tensor representing the operator A.
            state (torch.Tensor): Complex tensor representing the quantum state
                (either a state vector |ψ⟩ or density matrix ρ).

        Returns:
            torch.Tensor: Complex tensor representing the expectation value.
            
        Note:
            For pure states (vectors), computes the inner product ⟨ψ|A|ψ⟩.
            For mixed states (density matrices), computes the trace Tr(ρA).
        """
        # Handle pure states
        if len(state.shape) == 1:
            return torch.vdot(state, torch.mv(operator, state))
        # Handle density matrices
        else:
            return torch.trace(torch.matmul(operator, state))

    def tensor_product(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Calculate the tensor (Kronecker) product of two quantum operators or states.

        The tensor product is a fundamental operation in quantum mechanics that
        combines two separate quantum systems into a joint system.

        Args:
            A (torch.Tensor): First tensor (operator or state).
            B (torch.Tensor): Second tensor (operator or state).

        Returns:
            torch.Tensor: The tensor product A ⊗ B.
            
        Note:
            If A is m×n and B is p×q, the result will be (mp)×(nq).
            The operation preserves the type (complex/real) of the inputs.
        """
        return torch.kron(A, B)

    def displace(self, alpha: complex) -> Tensor:
        """
        Generate the displacement operator D(α) for a given complex amplitude.

        The displacement operator implements phase-space translations, creating
        coherent states when applied to the vacuum state.

        Args:
            alpha (complex): Complex amplitude representing the displacement in phase space.

        Returns:
            torch.Tensor: The displacement operator exp(αa† - α*a) as a complex matrix.
            
        Note:
            The displacement operator is unitary: D†(α)D(α) = I.
            It transforms the annihilation operator as: D†(α)aD(α) = a + α.
        """
        alpha = torch.complex(
            torch.tensor(float(alpha.real)), torch.tensor(float(alpha.imag))
        ).cuda()
        op = alpha * self.d.T.conj() - alpha.conj() * self.d
        return torch.matrix_exp(op)

    def squeeze(self, z: complex, offset: int = 0) -> Tensor:
        """
        Generate the single-mode squeezing operator S(z).

        The squeezing operator creates squeezed states by reducing quantum noise
        in one quadrature at the expense of increased noise in the conjugate quadrature.

        Args:
            z (complex): Squeezing parameter r*exp(iθ), where r is the squeezing
                magnitude and θ is the squeezing angle.
            offset (int, optional): The lowest number state included in the finite
                number state representation. Defaults to 0.

        Returns:
            torch.Tensor: The squeezing operator exp((z*a†a† - z*aa)/2) as a complex matrix.
            
        Note:
            The squeezing operator is unitary: S†(z)S(z) = I.
            It transforms the quadratures as:
            X → X*cosh(r) + P*sinh(r)
            P → P*cosh(r) - X*sinh(r)
            where r = |z| and θ = arg(z).
        """
        # Convert the squeezing parameter to a complex tensor on GPU
        z = torch.complex(
            torch.tensor(z.real, dtype=torch.float32),
            torch.tensor(z.imag, dtype=torch.float32),
        ).cuda()

        # Create the destruction operator squared
        asq = self.d @ self.d  # This is `destroy(N) ** 2` in qutip

        # Calculate the matrix exponent argument for squeezing
        op = 0.5 * torch.conj(z) * asq - 0.5 * z * asq.T.conj()

        # Matrix exponential to obtain the squeezing operator
        squeeze_op = torch.matrix_exp(op)

        return squeeze_op

    def partial_trace(self, rho: Tensor, mode: int) -> Tensor:
        """
        Compute the partial trace over one mode of a two-mode quantum state.

        The partial trace is used to obtain the reduced density matrix of a subsystem
        by tracing out the other subsystem.

        Args:
            rho (torch.Tensor): Two-mode density matrix of shape (N²×N²).
            mode (int): Which mode to trace out (0 or 1).

        Returns:
            torch.Tensor: The reduced density matrix of shape (N×N).
            
        Raises:
            ValueError: If mode is not 0 or 1.
            
        Note:
            For a two-mode state ρAB, the partial trace over B gives:
            ρA = TrB(ρAB) = ∑⟨i|B ρAB|i⟩B
            where {|i⟩B} is a basis for system B.
        """
        # Reshape to separate modes
        rho = rho.view(self.N, self.N, self.N, self.N)  # (i1, i2, j1, j2)

        # Trace out the specified mode by summing over indices
        if mode == 0:
            return torch.einsum("ijik->jk", rho)  # Trace over mode 0
        elif mode == 1:
            return torch.einsum("iijk->ik", rho)  # Trace over mode 1
        else:
            raise ValueError("Mode must be 0 or 1 for partial trace.")

    def fidelity(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Calculate the quantum fidelity between two quantum states.

        The fidelity F(ρ,σ) is a measure of the "closeness" of two quantum states ρ and σ.
        For pure states |ψ⟩ and |φ⟩, it reduces to |⟨ψ|φ⟩|².

        Args:
            A (torch.Tensor): First quantum state (pure state vector or density matrix).
            B (torch.Tensor): Second quantum state (pure state vector or density matrix).

        Returns:
            torch.Tensor: The fidelity value between 0 and 1.
            
        Note:
            - For pure states |ψ⟩ and |φ⟩: F = |⟨ψ|φ⟩|²
            - For mixed states ρ and σ: F = Tr[√(√ρσ√ρ)]²
            - Returns 0.0 if calculation fails or states are invalid
            - Handles NaN/inf values by attempting state recovery
            - Automatically normalizes input states
        """
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()

        # Check for NaN/inf values before proceeding
        if np.any(np.isnan(A_np)) or np.any(np.isinf(A_np)):
            logger.warning("NaN/inf detected in first state. Attempting recovery.")
            # Try to recover the state by removing tiny values
            A_np[np.abs(A_np) < 1e-10] = 0
            A_np[np.isnan(A_np)] = 0
            if np.any(np.isnan(A_np)) or np.any(np.isinf(A_np)):
                return torch.tensor(0.0, dtype=A.dtype, device=A.device)

        if np.any(np.isnan(B_np)) or np.any(np.isinf(B_np)):
            logger.warning("NaN/inf detected in second state. Attempting recovery.")
            B_np[np.abs(B_np) < 1e-10] = 0
            B_np[np.isnan(B_np)] = 0
            if np.any(np.isnan(B_np)) or np.any(np.isinf(B_np)):
                return torch.tensor(0.0, dtype=A.dtype, device=A.device)

        def normalize_state(state: np.ndarray) -> np.ndarray:
            """Normalize a quantum state vector or density matrix."""
            if state.ndim == 1:  # Pure state
                norm = np.sqrt(np.abs(np.sum(np.conj(state) * state)))
                if norm > 1e-10:  # Only normalize if norm is not too small
                    return state / norm
                return state
            else:  # Density matrix
                trace = np.trace(state)
                if abs(trace) > 1e-10:
                    return state / trace
                return state

        A_np = normalize_state(A_np)
        B_np = normalize_state(B_np)

        try:
            # Convert to QuTiP objects with error checking
            A_qt = Qobj(A_np)
            B_qt = Qobj(B_np)

            # Calculate fidelity with error handling
            try:
                fidelity_value = fidelity(A_qt, B_qt)
                # Check if result is valid
                if np.isnan(fidelity_value) or np.isinf(fidelity_value):
                    return torch.tensor(0.0, dtype=A.dtype, device=A.device)
                return torch.tensor(fidelity_value, dtype=A.dtype, device=A.device)
            except Exception as e:
                logger.warning(f"QuTiP fidelity calculation failed: {str(e)}")
                # Fallback calculation for pure states
                if A.dim() == 1 and B.dim() == 1:
                    overlap = np.abs(np.sum(np.conj(A_np) * B_np)) ** 2
                    return torch.tensor(overlap, dtype=A.dtype, device=A.device)
                return torch.tensor(0.0, dtype=A.dtype, device=A.device)

        except Exception as e:
            logger.error(f"Error in fidelity calculation: {str(e)}")
            return torch.tensor(0.0, dtype=A.dtype, device=A.device)

    def beam_splitter_qutip(
        self,
        one_in: Qobj,
        two_in: Qobj,
        theta: float = np.pi / 4
    ) -> Qobj:
        """
        Implement a beam splitter interaction using QuTiP.

        This is a reference implementation using QuTiP for verification purposes.
        For performance-critical applications, use the GPU-accelerated beam_splitter method.

        Args:
            one_in (Qobj): First input mode (ket or density matrix).
            two_in (Qobj): Second input mode (ket or density matrix).
            theta (float, optional): Beam splitter mixing angle in radians. Defaults to π/4.

        Returns:
            Qobj: Two-mode output density matrix.
            
        Note:
            The beam splitter transformation is:
            a₁ → a₁cos(θ) - a₂sin(θ)
            a₂ → a₁sin(θ) + a₂cos(θ)
            where a₁, a₂ are the mode operators.
        """
        # Convert inputs to density matrices if needed
        one_dm = ket2dm(one_in) if one_in.type == "ket" else one_in
        two_dm = ket2dm(two_in) if two_in.type == "ket" else two_in

        # Create destruction operators using QutIP's tensor
        destroy_one = tensor(destroy(self.N), identity(self.N))
        destroy_two = tensor(identity(self.N), destroy(self.N))

        # Compute the unitary operator
        generator = -theta * (
            destroy_one.dag() * destroy_two - destroy_two.dag() * destroy_one
        )
        unitary = generator.expm()

        # Compute final state
        input_dm = tensor(one_dm, two_dm)
        return unitary.dag() * input_dm * unitary

    def measure_mode_qutip(
        self,
        two_mode_dm: Qobj,
        projector: Qobj,
        mode: int
    ) -> Qobj:
        """
        Perform a projective measurement on a two-mode state using QuTiP.

        This is a reference implementation using QuTiP for verification purposes.
        For performance-critical applications, use the GPU-accelerated measure_mode method.

        Args:
            two_mode_dm (Qobj): Two-mode quantum state density operator.
            projector (Qobj): Measurement projection operator.
            mode (int): Which mode to measure (1 or 2).

        Returns:
            Qobj: Normalized post-measurement state in the untouched mode.
            
        Raises:
            ValueError: If mode is not 1 or 2.
            
        Note:
            The measurement process:
            1. Applies the projector to the specified mode
            2. Traces out the measured mode
            3. Normalizes the resulting state
        """
        if mode not in [1, 2]:
            raise ValueError("Mode must be 1 or 2.")

        id_N = identity(self.N)

        # Pre-compute measurement operator
        measurement = tensor(projector, id_N) if mode == 1 else tensor(id_N, projector)

        # Perform measurement and partial trace
        measured_state = two_mode_dm * measurement
        traced_state = measured_state.ptrace(1 if mode == 1 else 0)

        return traced_state.unit()

    def verify_catfid(self, num_states: int = 10) -> None:
        """
        Verify the consistency between QuTiP and GPU implementations of cat state fidelity.

        This method generates random test states and compares the fidelity calculations
        between the QuTiP-based and GPU-accelerated implementations to ensure they
        produce consistent results within numerical precision.

        Args:
            num_states (int, optional): Number of random test states to generate. Defaults to 10.
            
        Note:
            - Prints comparison results and statistics
            - Useful for debugging and validation
            - Does not return a value but raises warnings if discrepancies are found
        """
        # Generate random states
        states = []
        for _ in range(num_states):
            # Generate random complex amplitudes
            amplitudes = np.random.rand(self.N) + 1j * np.random.rand(self.N)
            amplitudes /= np.linalg.norm(amplitudes)
            states.append(torch.tensor(amplitudes, dtype=torch.complex64).cuda())

        # Parameters for catfid calculation
        u = 3
        c = 1
        k = 100
        projector = p0_projector(self.N)
        projector_gpu = torch.tensor(projector.full(), dtype=torch.complex64).cuda()

        for state in states:
            # Calculate catfid using QuTiP
            state_qobj = Qobj(state.detach().cpu().numpy())
            print(f"State: {state_qobj}")
            cat_sq_qutip, out_fid_qutip = catfid(self.N, state_qobj, u, c, k, projector)

            # Calculate catfid using PyTorch-based GPU implementation
            cat_sq_torch, out_fid_torch = self.catfid_gpu(state, u, c, k, projector_gpu)

            # Compare the results
            print(f"State: {state}")
            print(
                f"Cat Squeezing: QuTiP = {cat_sq_qutip:.4f}, PyTorch = {cat_sq_torch:.4f}"
            )
            print(
                f"Output Fidelity: QuTiP = {out_fid_qutip:.4f}, PyTorch = {out_fid_torch:.4f}"
            )

            # Check if the results are close
            assert np.allclose(cat_sq_qutip, cat_sq_torch.item(), atol=1e-4)
            assert np.allclose(out_fid_qutip, out_fid_torch.item(), atol=1e-4)

        print("Verification successful!")

    def catfid_gpu(
        self,
        state: Tensor,
        u: float,
        parity: str,
        c: int,
        k: int,
        projector: Tensor,
        operator: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute cat squeezing and output fidelity for a quantum state using GPU acceleration.

        This method calculates two key metrics for cat state preparation:
        1. Cat squeezing: A measure of how well the state approximates an ideal cat state
        2. Output fidelity: The overlap between the measured state and an ideal squeezed cat state

        Args:
            state (torch.Tensor): Input quantum state (normalized state vector).
            u (float): Displacement amplitude for the ideal cat state.
            parity (str): Type of cat state to compare against ("even" or "odd").
            c (int): Parameter for operator generation (code distance).
            k (int): Parameter for operator generation (Fock space cutoff).
            projector (torch.Tensor): Measurement projector.
            operator (torch.Tensor, optional): Pre-computed squeezing operator.
                If None, generated internally. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cat squeezing and output fidelity values.
            
        Note:
            - Falls back to QuTiP implementation if numerical instabilities occur
            - The ideal state is constructed as (D(u/2) ± D(-u/2))S(r)|0⟩
              where D is displacement, S is squeezing, and ± depends on parity
            - All operations are performed on GPU for efficiency
        """
        if operator is None:
            operator = self.operator_new_gpu(u, 0, c, k)

        # Calculate cat squeezing
        cat_squeezing = self.expect(operator, state)

        # Create ideal state
        vacuum = torch.zeros(self.N, dtype=torch.complex64).cuda()
        vacuum[0] = 1.0

        # Apply operations for ideal state
        squeezed = torch.mv(self.squeeze(np.log(2) / 2), vacuum)
        displaced_plus = torch.mv(self.displace(u / 2), squeezed)
        displaced_minus = torch.mv(self.displace(-u / 2), squeezed)
        if parity == "even":
            ideal_state = displaced_plus + displaced_minus
        elif parity == "odd":
            ideal_state = displaced_plus - displaced_minus
        ideal_state = ideal_state / torch.sqrt(torch.vdot(ideal_state, ideal_state))

        # Calculate output state
        output_state = self.measure_mode(
            self.beam_splitter(state, vacuum), projector, 1
        )

        # Check for NaN or inf in output_state
        if torch.any(torch.isnan(output_state)) or torch.any(torch.isinf(output_state)):
            logger.warning("Numerical instability detected, falling back to QuTiP implementation.")
            state_qobj = Qobj(state.detach().cpu().numpy())
            output_state_qobj = self.measure_mode_qutip(
                self.beam_splitter_qutip(state_qobj, basis(self.N, 0)),
                p0_projector(self.N),
                1,
            )
            output_state = torch.tensor(
                output_state_qobj.full(), dtype=torch.complex64
            ).cuda()

        # Calculate fidelity
        output_fidelity = self.fidelity(output_state, ideal_state)

        return cat_squeezing, output_fidelity

    def gkp_operator_gpu(self) -> Tensor:
        """
        Generate the GKP (Gottesman-Kitaev-Preskill) operator on GPU.

        The GKP operator is used to measure the quality of GKP state preparation.
        It is constructed to detect the characteristic properties of grid states.

        Returns:
            torch.Tensor: GKP operator matrix as a complex tensor on GPU.
            
        Note:
            Uses QuTiP's gkp_operator_new function for initial construction,
            then converts to GPU tensor for efficient computation.
        """
        op = gkp_operator_new(self.N)
        return torch.tensor(op.full(), dtype=torch.complex64).cuda()

    def gkp_operator_odd_gpu(self) -> Tensor:
        """
        Generate the odd-parity GKP operator on GPU.

        This is a placeholder implementation that currently returns the same
        operator as gkp_operator_gpu(). A proper implementation should generate
        an operator specific to odd-parity GKP states.

        Returns:
            torch.Tensor: Odd-parity GKP operator matrix as a complex tensor on GPU.
            
        Note:
            TODO: Implement proper odd-parity GKP operator construction.
        """
        op = gkp_operator_new(self.N)
        return torch.tensor(op.full(), dtype=torch.complex64).cuda()

    def catgkp_gpu(
        self,
        state: Tensor,
        rounds: int,
        c: int,
        k: int,
        projector: Tensor,
        operator: Optional[Tensor] = None,
        gkp_operator: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate cat and GKP squeezing metrics for a quantum state.

        This method evaluates how well a state approximates both a cat state and
        a GKP state by computing their respective squeezing parameters.

        Args:
            state (torch.Tensor): Input quantum state (normalized state vector).
            rounds (int): Number of breeding protocol iterations.
            c (int): Code distance parameter.
            k (int): Fock space cutoff parameter.
            projector (torch.Tensor): Measurement projector.
            operator (torch.Tensor, optional): Pre-computed cat squeezing operator.
                If None, generated internally. Defaults to None.
            gkp_operator (torch.Tensor, optional): Pre-computed GKP operator.
                If None, generated internally. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cat squeezing and GKP squeezing values.
            
        Note:
            - The displacement amplitude u is calculated based on the number of rounds
            - Uses breeding protocol to transform the input state
            - Falls back to QuTiP implementation if numerical instabilities occur
        """
        u = 2 * np.sqrt(2) * np.sqrt(np.pi) * 2 ** ((rounds - 3) / 2)

        if operator == None:
            operator = self.operator_new_gpu(u, 0, c, k)

        if gkp_operator == None:
            gkp_operator = self.gkp_operator_gpu()

        # Calculate cat squeezing
        cat_squeezing = self.expect(operator, state)

        # Calculate output state
        output_state = self.breeding_gpu(rounds, state, projector)

        # Check for NaN or inf in output_state
        if torch.any(torch.isnan(output_state)) or torch.any(torch.isinf(output_state)):
            # Fallback to using qutip functions
            print("Falling back to QuTiP functions.")
            state_qobj = Qobj(state.detach().cpu().numpy())
            output_state_qobj = self.measure_mode_qutip(
                self.beam_splitter_qutip(state_qobj, basis(self.N, 0)),
                p0_projector(self.N),
                1,
            )
            output_state = torch.tensor(
                output_state_qobj.full(), dtype=torch.complex64
            ).cuda()

        # Calculate gkp squeezingself.N, self.u, 0, self.c, self.k
        gkp_squeezing = self.expect(gkp_operator, output_state)

        return cat_squeezing, gkp_squeezing

    def verify_catgkp(self, num_states: int = 10) -> None:
        """
        Verify consistency between QuTiP and GPU implementations of cat-GKP metrics.

        This method generates random test states and compares the cat squeezing and
        GKP squeezing calculations between the QuTiP-based and GPU-accelerated
        implementations to ensure they produce consistent results.

        Args:
            num_states (int, optional): Number of random test states to generate. Defaults to 10.
            
        Note:
            - Prints comparison results and statistics
            - Raises AssertionError if results don't match within tolerance
            - Uses atol=1e-4 for numerical comparisons
        """
        # Generate random states
        states = []
        for _ in range(num_states):
            # Generate random complex amplitudes
            amplitudes = np.random.rand(self.N) + 1j * np.random.rand(self.N)
            amplitudes /= np.linalg.norm(amplitudes)
            states.append(torch.tensor(amplitudes, dtype=torch.complex64).cuda())

        # Parameters for catfid calculation
        rounds = 3
        c = 1
        k = 100
        projector = p0_projector(self.N)
        projector_gpu = torch.tensor(projector.full(), dtype=torch.complex64).cuda()

        for state in states:
            # Calculate catfid using QuTiP
            state_qobj = Qobj(state.detach().cpu().numpy())
            logger.debug(f"Testing state: {state_qobj}")
            cat_sq_qutip, gkp_sq_qutip = catgkp(
                self.N, state_qobj, rounds, c, k, projector
            )

            # Calculate catfid using PyTorch-based GPU implementation
            cat_sq_torch, gkp_sq_torch = self.catgkp_gpu(
                state, rounds, c, k, projector_gpu
            )

            # Compare the results
            logger.info(
                f"Cat Squeezing: QuTiP = {cat_sq_qutip:.4f}, PyTorch = {cat_sq_torch:.4f}"
            )
            logger.info(
                f"GKP Squeezing: QuTiP = {gkp_sq_qutip:.4f}, PyTorch = {gkp_sq_torch:.4f}"
            )

            # Check if the results are close
            assert np.allclose(cat_sq_qutip, cat_sq_torch.item(), atol=1e-4)
            assert np.allclose(gkp_sq_qutip, gkp_sq_torch.item(), atol=1e-4)

        logger.info("Verification successful!")

    def state_to_params_gpu(self, state: Tensor) -> Tensor:
        """
        Convert a quantum state tensor to optimization parameters.

        This method splits a complex quantum state into its real and imaginary parts
        for use in optimization algorithms that work with real-valued parameters.

        Args:
            state (torch.Tensor): Normalized quantum state tensor.

        Returns:
            torch.Tensor: Parameter tensor with concatenated real and imaginary parts.
            
        Note:
            The output tensor has twice the length of the input, with real parts
            followed by imaginary parts.
        """
        real_parts = state.real
        imag_parts = state.imag
        params = torch.cat([real_parts, imag_parts], dim=1)
        return params

    def state_to_qutip(self, state: Tensor) -> Qobj:
        """
        Convert a PyTorch quantum state tensor to a QuTiP Qobj.

        This method facilitates interoperability between PyTorch and QuTiP by
        converting GPU tensors to QuTiP quantum objects.

        Args:
            state (torch.Tensor): Normalized quantum state tensor.

        Returns:
            qutip.Qobj: Normalized QuTiP quantum object representing the state.
            
        Note:
            The state is automatically normalized using QuTiP's unit() method.
        """
        real_parts = state.real.cpu().numpy()
        imag_parts = state.imag.cpu().numpy()
        qstate = Qobj(real_parts + 1j * imag_parts)
        return qstate.unit()

    def params_to_qutip(self, params: np.ndarray) -> Qobj:
        """
        Convert optimization parameters to a QuTiP quantum object.

        This method reconstructs a quantum state from its real and imaginary parts
        and returns it as a normalized QuTiP quantum object.

        Args:
            params (numpy.ndarray): Parameter array with concatenated real and
                imaginary parts.

        Returns:
            qutip.Qobj: Normalized QuTiP quantum object representing the state.
            
        Note:
            Assumes params is structured as [real_parts, imag_parts] with equal lengths.
        """
        N = params.shape[0] // 2
        real_parts = params[:N]
        imag_parts = params[N:]
        qstate = Qobj(real_parts + 1j * imag_parts)
        return qstate.unit()

    def params_to_state_gpu(self, params: np.ndarray) -> Tensor:
        """
        Convert optimization parameters to a normalized quantum state on GPU.

        This method reconstructs a quantum state from its real and imaginary parts
        and returns it as a normalized PyTorch tensor on the GPU.

        Args:
            params (numpy.ndarray): Parameter array with concatenated real and
                imaginary parts.

        Returns:
            torch.Tensor: Normalized quantum state tensor on GPU.
            
        Note:
            - Assumes params is structured as [real_parts, imag_parts] with equal lengths
            - Automatically moves the state to GPU and normalizes it
        """
        # Split into real and imaginary parts
        real_parts = torch.tensor(params[: params.shape[0] // 2]).cuda()
        imag_parts = torch.tensor(params[params.shape[0] // 2 :]).cuda()

        # Create complex states using real-valued tensors
        state = torch.complex(real_parts.float(), imag_parts.float()).cuda()

        logger.debug(f"Created state from parameters: {state}")

        # Normalize states
        norms = torch.sqrt(torch.sum(torch.abs(state) ** 2))
        state = state / norms

        return state

    def operator_new_gpu(
        self,
        u: float,
        phi: float,
        c: float,
        k: int
    ) -> Tensor:
        """
        Generate a quantum operator and convert it to a GPU tensor.

        This method creates a high-dimensional operator matrix using QuTiP's
        operator_new function and converts it to a GPU tensor for efficient
        computation.

        Args:
            u (float): Displacement amplitude for the ideal state.
            phi (float): Phase parameter for operator generation.
            c (float): Code distance parameter.
            k (int): Fock space cutoff parameter.

        Returns:
            torch.Tensor: Operator matrix as a complex tensor on GPU.
            
        Note:
            Uses QuTiP's operator_new function for initial construction,
            then converts to GPU tensor for efficient computation.
        """
        op = operator_new(self.N, u, phi, c, k)
        return torch.tensor(op.full(), dtype=torch.complex64).cuda()


if __name__ == "__main__":
    quantum_ops = GPUQuantumOps(30)
    quantum_ops.verify_catfid(num_states=10)
    quantum_ops.verify_catgkp(num_states=10)
