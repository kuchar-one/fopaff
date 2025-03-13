import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cuda_helpers import DeviceMonitor
import torch
import math
import numpy as np
from typing import Optional, Tuple
from torch import Tensor
from qutip import Qobj, destroy, identity, basis, tensor, fidelity, ket2dm, wigner
from src.qutip_quantum_ops import (
    catfid,
    operator_new,
    gkp_operator_new,
    p0_projector,
    catgkp,
)
from src.logger import setup_logger

logger = setup_logger()


class GPUQuantumOps:
    def __init__(self, N):
        """
        Initialize GPUQuantumOps object with a given Hilbert space dimension N.

        This creates a DeviceMonitor to track CUDA device usage and prints
        information about the currently selected device. It also pre-computes
        operators on the default device (cuda:0) which will be moved to the
        correct device when using the GPUDeviceWrapper.
        """
        self.N = N
        self.monitor = DeviceMonitor()
        self.device = "cuda:0"  # Default device, will be updated by wrapper

        # Print initial CUDA information
        self.monitor.print_cuda_info()
        print("\n")

        # Pre-compute operators on default device (will be moved by wrapper)
        self.d = self._create_destroy_operator().to(self.device)
        self.identity = torch.eye(N, dtype=torch.complex64).to(self.device)

    def _ensure_same_device(self, *tensors):
        """
        Ensure that all provided tensors are on the same device.

        Args:
            *tensors: Variable length argument list of tensors to check.

        Returns:
            torch.device: The common device of the tensors, or the default device if no tensors are provided.

        Raises:
            RuntimeError: If tensors are found on different devices.
        """

        devices = {t.device for t in tensors if torch.is_tensor(t)}
        if len(devices) > 1:
            raise RuntimeError(f"Tensors found on different devices: {devices}")
        return list(devices)[0] if devices else self.device

    def beam_splitter(self, one_in, two_in, theta=np.pi / 4):
        """
        Simulate a beam splitter operation on two input quantum states.

        This function applies a beam splitter transformation to two input
        density matrices, `one_in` and `two_in`, using a specified angle `theta`.

        Args:
            one_in (torch.Tensor): The first input quantum state, represented as a
                density matrix or a state vector.
            two_in (torch.Tensor): The second input quantum state, represented as a
                density matrix or a state vector.
            theta (float, optional): The angle of the beam splitter in radians. Default
                is π/4.

        Returns:
            torch.Tensor: The resulting quantum state after applying the beam splitter
            transformation, as a density matrix.
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

            # Rest of the method remains the same but uses self.device explicitly
            d1 = self.tensor_product(self.d, self.identity)
            d2 = self.tensor_product(self.identity, self.d)

            op = -theta * (
                torch.matmul(d1.T.conj(), d2) - torch.matmul(d2.T.conj(), d1)
            )

            try:
                U = torch.matrix_exp(op)
                torch.save(U.cpu(), cache_path)
            except Exception:
                print("Warning: Matrix exponential failed, using fallback method")
                U = torch.eye(op.shape[0], dtype=op.dtype, device=self.device)
                op_power = torch.eye(op.shape[0], dtype=op.dtype, device=self.device)
                for i in range(1, 10):
                    op_power = torch.matmul(op_power, op) / i
                    U = U + op_power

        if len(one_in.shape) == 1:
            one_in = torch.outer(one_in, one_in.conj())
        if len(two_in.shape) == 1:
            two_in = torch.outer(two_in, two_in.conj())

        input_dm = self.tensor_product(one_in, two_in)
        result = torch.matmul(torch.matmul(U.T.conj(), input_dm), U)

        return result.to(self.device)

    def breeding_gpu(self, rounds, input_state, projector):
        """
        Simulate the breeding protocol to generate a GKP state.

        This function applies the breeding protocol to an input quantum state
        `input_state` and a measurement projector `projector`, repeating the
        process `rounds` times. The resulting quantum state is returned as a
        density matrix.

        Args:
            rounds (int): The number of breeding rounds.
            input_state (torch.Tensor): The input quantum state, represented as a
                density matrix or a state vector.
            projector (torch.Tensor): The measurement projector, represented as a
                matrix.

        Returns:
            torch.Tensor: The resulting quantum state after applying the breeding
            protocol, as a density matrix.
        """
        input_state = input_state.to(self.device)
        projector = projector.to(self.device)

        device = self._ensure_same_device(input_state, projector)
        # logger.info(f"breeding_gpu: Ensuring tensors are on device {device}")
        # logger.info(f"Input state device: {input_state.device}")
        # logger.info(f"Projector device: {projector.device}")

        input_state = input_state.to(device)
        projector = projector.to(device)

        # logger.info(f"Input state moved to device: {input_state.device}")
        # logger.info(f"Projector moved to device: {projector.device}")

        if rounds == 0:
            return input_state

        temp = self.beam_splitter(input_state, input_state)
        new_state = self.measure_mode(temp, projector, 1)
        output_state = self.breeding_gpu(rounds - 1, new_state, projector)

        return output_state.to(self.device)

    def measure_mode(self, two_mode_dm, projector, mode):
        """
        Perform a measurement on the given two-mode density matrix using the specified projector.

        Args:
            two_mode_dm (torch.Tensor): The two-mode density matrix to be measured.
            projector (torch.Tensor): The projector used to measure the first mode.
            mode (int): The mode to be measured. Must be 1 or 2.

        Returns:
            torch.Tensor: The resulting density matrix after measurement.
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

    def get_free_memory(self):
        """
        Get the amount of free memory in the current PyTorch device in MB.

        Returns:
            int: The amount of free memory in MB.
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)  # Convert bytes to MB
        return 0

    def _create_destroy_operator(self, offset=0):
        """
        Create the destruction (lowering) operator in PyTorch for N dimensions.

        Parameters
        ----------
        N : int
            Number of basis states in the Hilbert space.
        offset : int, default: 0
            The lowest number state that is included in the finite number state
            representation of the operator.

        Returns
        -------
        d : torch.Tensor
            The lowering operator as a complex matrix.
        """
        # Initialize the destruction operator matrix with zeros on the GPU
        d = torch.zeros((self.N, self.N), dtype=torch.complex64).cuda()

        # Populate the off-diagonal elements with the square roots of integers
        for n in range(1, self.N):
            d[n - 1, n] = torch.sqrt(torch.tensor(n + offset, dtype=torch.float32)).to(
                d.device
            )

        return d

    def expect(self, operator, state):
        """Calculate expectation value of an operator with respect to a state

        Args:
            operator: Complex tensor representing the operator
            state: Complex tensor representing the quantum state

        Returns:
            Complex tensor representing expectation value <ψ|A|ψ>
        """
        # Handle pure states
        if len(state.shape) == 1:
            return torch.vdot(state, torch.mv(operator, state))
        # Handle density matrices
        else:
            return torch.trace(torch.matmul(operator, state))

    def tensor_product(self, A, B):
        """
        Calculate the tensor product of two PyTorch tensors.

        Parameters
        ----------
        A : torch.Tensor
            First tensor to calculate the tensor product with.
        B : torch.Tensor
            Second tensor to calculate the tensor product with.

        Returns
        -------
        result : torch.Tensor
            The tensor product of A and B.
        """
        return torch.kron(A, B)

    def displace(self, alpha):
        """
        Generate the displacement operator for a given complex amplitude.

        This function creates a displacement operator, which is a fundamental
        operation in quantum optics, corresponding to a shift in phase space.

        Parameters
        ----------
        alpha : complex
            The complex amplitude representing the displacement in phase space.

        Returns
        -------
        torch.Tensor
            The displacement operator as a complex matrix on the GPU.
        """

        alpha = torch.complex(
            torch.tensor(float(alpha.real)), torch.tensor(float(alpha.imag))
        ).cuda()
        op = alpha * self.d.T.conj() - alpha.conj() * self.d
        return torch.matrix_exp(op)

    def squeeze(self, z, offset=0):
        """Generate the single-mode squeezing operator on GPU with PyTorch.

        Parameters
        ----------
        N : int
            Dimension of the Hilbert space.
        z : complex
            Squeezing parameter.
        offset : int, default=0
            The lowest number state that is included in the finite number state
            representation of the operator.

        Returns
        -------
        squeeze_op : torch.Tensor
            The squeezing operator as a complex matrix.
        """
        # Convert the squeezing parameter `z` to a complex tensor on GPU.
        z = torch.complex(
            torch.tensor(z.real, dtype=torch.float32),
            torch.tensor(z.imag, dtype=torch.float32),
        ).cuda()

        # Create the destruction operator squared, `asq`.
        asq = self.d @ self.d  # This is `destroy(N) ** 2` in qutip

        # Calculate the matrix exponent argument for squeezing
        op = 0.5 * torch.conj(z) * asq - 0.5 * z * asq.T.conj()

        # Matrix exponential to obtain the squeezing operator
        squeeze_op = torch.matrix_exp(op)

        return squeeze_op

    def partial_trace(self, rho, mode):
        """
        Compute partial trace over one mode in a two-mode density matrix.

        Args:
            rho (torch.Tensor): Two-mode density matrix.
            N (int): Dimension of truncated Fock space.
            mode (int): Mode to trace out (0 or 1).

        Returns:
            torch.Tensor: Resulting density matrix after tracing out specified mode.
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

    def fidelity(self, A, B):
        """
        Calculate the fidelity between two states.

        The fidelity is a measure of the closeness of two quantum states.
        It is defined as the square of the absolute value of the overlap
        between the two states.

        Parameters
        ----------
        A : torch.Tensor
            First state.
        B : torch.Tensor
            Second state.

        Returns
        -------
        fidelity_value : torch.Tensor
            The fidelity between the two states.

        Notes
        -----
        This function first converts the input tensors to numpy arrays
        and checks for NaN/inf values. If any are found, it attempts to
        recover the state by removing tiny values. If the state is still
        invalid after recovery, the function returns 0.0.

        The function then normalizes the states using the `normalize_state`
        function and calculates the fidelity using QuTiP's fidelity function.
        If the result is invalid, it falls back to a pure state calculation
        if the input tensors are both 1D (i.e. pure states).

        If any errors occur during the calculation, the function returns 0.0.
        """
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()

        # Check for NaN/inf values before proceeding
        if np.any(np.isnan(A_np)) or np.any(np.isinf(A_np)):
            print("Warning: NaN/inf detected in first state. Attempting recovery.")
            # Try to recover the state by removing tiny values
            A_np[np.abs(A_np) < 1e-10] = 0
            A_np[np.isnan(A_np)] = 0
            if np.any(np.isnan(A_np)) or np.any(np.isinf(A_np)):
                return torch.tensor(0.0, dtype=A.dtype, device=A.device)

        if np.any(np.isnan(B_np)) or np.any(np.isinf(B_np)):
            print("Warning: NaN/inf detected in second state. Attempting recovery.")
            B_np[np.abs(B_np) < 1e-10] = 0
            B_np[np.isnan(B_np)] = 0
            if np.any(np.isnan(B_np)) or np.any(np.isinf(B_np)):
                return torch.tensor(0.0, dtype=A.dtype, device=A.device)

        # Ensure proper normalization
        def normalize_state(state):
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
            except:
                # Fallback calculation for pure states
                if A.dim() == 1 and B.dim() == 1:
                    overlap = np.abs(np.sum(np.conj(A_np) * B_np)) ** 2
                    return torch.tensor(overlap, dtype=A.dtype, device=A.device)
                return torch.tensor(0.0, dtype=A.dtype, device=A.device)

        except Exception as e:
            print(f"Error in fidelity calculation: {str(e)}")
            return torch.tensor(0.0, dtype=A.dtype, device=A.device)

    def beam_splitter_qutip(self, one_in, two_in, theta=np.pi / 4):
        """Optimized beam splitter interaction

        Args:
            N (int): truncated Fock space dimension
            one_in (ket or oper): first mode input
            two_in (ket or oper): second mode input
            theta (float, optional): BS parameter, balanced default

        Returns:
            oper: two-mode output density matrix
        """
        N = self.N

        # Convert inputs to density matrices if needed
        one_dm = ket2dm(one_in) if one_in.type == "ket" else one_in
        two_dm = ket2dm(two_in) if two_in.type == "ket" else two_in

        # Create destruction operators using QutIP's tensor
        destroy_one = tensor(destroy(N), identity(N))
        destroy_two = tensor(identity(N), destroy(N))

        # Compute the unitary operator
        generator = -theta * (
            destroy_one.dag() * destroy_two - destroy_two.dag() * destroy_one
        )
        unitary = generator.expm()

        # Compute final state
        input_dm = tensor(one_dm, two_dm)
        return unitary.dag() * input_dm * unitary

    def measure_mode_qutip(self, two_mode_dm, projector, mode):
        """Optimized projection measurement of two mode state

        Args:
            N (int): truncated Fock space dimension
            two_mode_dm (oper): two mode state density operator
            projector (oper): measurement projection
            mode (1 or 2): mode number

        Returns:
            oper: conditional output density matrix in untouched mode
        """
        N = self.N

        id_N = identity(N)

        # Pre-compute measurement operator
        measurement = tensor(projector, id_N) if mode == 1 else tensor(id_N, projector)

        # Perform measurement and partial trace
        measured_state = two_mode_dm * measurement
        traced_state = measured_state.ptrace(1 if mode == 1 else 0)

        return traced_state.unit()

    def verify_catfid(self, num_states=10):
        """
        Verify that the catfid calculations are consistent between QuTiP and PyTorch-based GPU implementations.

        Args:
            num_states (int): Number of random states to generate for the verification.
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

    def catfid_gpu(self, state, u, parity, c, k, projector, operator=None):
        """
        Compute the cat squeezing and output fidelity for a given quantum state using GPU acceleration.

        This function calculates the cat squeezing and the fidelity between an evolved quantum
        state and an ideal squeezed Schrödinger cat state. It leverages GPU-accelerated operations
        to perform the calculations efficiently.

        Parameters
        ----------
        state : torch.Tensor
            The input quantum state represented as a complex-valued tensor.
        u : float
            Displacement amplitude used in the preparation of the ideal state.
        parity : str
            Specifies the parity of the ideal state ("even" or "odd").
        c : int
            A parameter used in the operator generation.
        k : int
            A parameter used in the operator generation.
        projector : torch.Tensor
            The projector used in the measurement step.
        operator : torch.Tensor, optional
            Pre-computed operator for squeezing calculation. If not provided, it is generated internally.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the calculated cat squeezing and the output fidelity. Both are returned
            as complex-valued tensors.

        Notes
        -----
        The function falls back to using QuTiP functions if invalid (NaN or inf) values are detected
        in the output state during computation.
        """

        if operator == None:
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

        # Calculate fidelity
        output_fidelity = self.fidelity(output_state, ideal_state)

        return cat_squeezing, output_fidelity

    def gkp_operator_gpu(self):
        """
        Return the GKP operator as a GPU tensor.

        This function generates the GKP operator matrix using QuTiP and then
        converts it to a PyTorch tensor, which is then moved to the GPU.

        Returns
        -------
        torch.Tensor
            The GKP operator matrix as a complex-valued tensor on the GPU.
        """
        op = gkp_operator_new(self.N)
        return torch.tensor(op.full(), dtype=torch.complex64).cuda()

    # TODO: actually implement this
    def gkp_operator_odd_gpu(self):
        op = gkp_operator_new(self.N)
        return torch.tensor(op.full(), dtype=torch.complex64).cuda()

    def catgkp_gpu(
        self, state, rounds, c, k, projector, operator=None, gkp_operator=None
    ):
        """
        Calculate cat squeezing and GKP squeezing of a given state.

        Args:
            state (torch.Tensor): The input state as a complex-valued tensor.
            rounds (int): The number of rounds of the breeding protocol.
            c (int): The code distance of the GKP code.
            k (int): The number of Fock states to include in the GKP code.
            projector (torch.Tensor): The projector onto the first Fock state.
            operator (torch.Tensor, optional): The operator to use for calculating cat squeezing.
                If not provided, the default operator is used.
            gkp_operator (torch.Tensor, optional): The GKP operator to use for calculating GKP squeezing.
                If not provided, the default GKP operator is used.

        Returns:
            tuple: The cat squeezing and GKP squeezing as a tuple.
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

    def verify_catgkp(self, num_states=10):
        """
        Verify that the catfid calculations are consistent between QuTiP and PyTorch-based GPU implementations.

        Args:
            num_states (int): Number of random states to generate for the verification.
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
            print(f"State: {state_qobj}")
            cat_sq_qutip, gkp_sq_qutip = catgkp(
                self.N, state_qobj, rounds, c, k, projector
            )

            # Calculate catfid using PyTorch-based GPU implementation
            cat_sq_torch, gkp_sq_torch = self.catgkp_gpu(
                state, rounds, c, k, projector_gpu
            )

            # Compare the results
            print(f"State: {state}")
            print(
                f"Cat Squeezing: QuTiP = {cat_sq_qutip:.4f}, PyTorch = {cat_sq_torch:.4f}"
            )
            print(
                f"GKP Squeezing: QuTiP = {gkp_sq_qutip:.4f}, PyTorch = {gkp_sq_torch:.4f}"
            )

            # Check if the results are close
            assert np.allclose(cat_sq_qutip, cat_sq_torch.item(), atol=1e-4)
            assert np.allclose(gkp_sq_qutip, gkp_sq_torch.item(), atol=1e-4)

        print("Verification successful!")

    def state_to_params_gpu(self, state):
        """
        Converts a quantum state tensor to a parameters tensor.

        Args:
            state (torch.Tensor): Normalized quantum state tensor
        Returns:
            torch.Tensor: Parameters tensor with real and imaginary parts
        """
        real_parts = state.real
        imag_parts = state.imag
        params = torch.cat([real_parts, imag_parts], dim=1)
        return params

    def state_to_qutip(self, state):
        """
        Converts a PyTorch quantum state tensor to a Qutip Quantum object.

        Args:
            state (torch.Tensor): Normalized quantum state tensor
        Returns:
            qutip.Qobj: Qutip Quantum object representing the state
        """
        real_parts = state.real.numpy()
        imag_parts = state.imag.numpy()
        qstate = Qobj(real_parts + 1j * imag_parts)
        return qstate.unit()

    def params_to_qutip(self, params):
        """
        Converts a parameters tensor to a Qutip Quantum object.

        Args:
            params array: Parameters tensor with real and imaginary parts
        Returns:
            qutip.Qobj: Qutip Quantum object representing the state
        """
        N = params.shape[0] // 2
        real_parts = params[:N]
        imag_parts = params[N:]
        qstate = Qobj(real_parts + 1j * imag_parts)
        return qstate.unit()

    def params_to_state_gpu(self, params):
        """
        Converts parameters tensor to normalized quantum states tensor.

        Args:
            params (torch.Tensor): Input parameters tensor
        Returns:
            torch.Tensor: Normalized quantum states tensor
        """
        # Split into real and imaginary parts
        real_parts = torch.tensor(params[: params.shape[0] // 2]).cuda()
        imag_parts = torch.tensor(params[params.shape[0] // 2 :]).cuda()

        # Create complex states using real-valued tensors
        state = torch.complex(real_parts.float(), imag_parts.float()).cuda()

        print(state)

        # Normalize states
        norms = torch.sqrt(torch.sum(torch.abs(state) ** 2))
        state = state / norms

        return state

    def operator_new_gpu(self, u, phi, c, k):
        """
        Generate a high-dimensional operator matrix and convert it to a GPU tensor.

        The operator is generated using the :func:`operator_new` function from
        :mod:`qutip_quantum_ops` and converted to a GPU tensor using PyTorch.

        Parameters
        ----------
        u : float
            Displacement amplitude used in the preparation of the ideal state.
        phi : float
            A parameter used in the generation of the operator.
        c : float
            A parameter used in the generation of the operator.
        k : int
            A parameter used in the generation of the operator.

        Returns
        -------
        torch.Tensor
            The operator matrix as a GPU tensor.
        """
        op = operator_new(self.N, u, phi, c, k)
        # Convert to GPU tensor
        return torch.tensor(op.full(), dtype=torch.complex64).cuda()


if __name__ == "__main__":
    quantum_ops = GPUQuantumOps(30)
    quantum_ops.verify_catfid(num_states=10)
    quantum_ops.verify_catgkp(num_states=10)
