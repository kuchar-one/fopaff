import torch
import time
from functools import wraps
from typing import Union, Callable, Any


class DeviceMonitor:
    """
    A utility class for monitoring and debugging PyTorch tensor device operations.

    This class provides static methods for tracking tensor device locations,
    monitoring CUDA device usage, and profiling tensor operations. It is particularly
    useful for debugging device-related issues and performance monitoring in
    GPU-accelerated applications.

    The class offers functionality to:
    - Track tensor device locations
    - Monitor CUDA device memory usage
    - Profile function execution times
    - Debug device-related issues

    Example:
        >>> monitor = DeviceMonitor()
        >>> tensor = torch.randn(10, device='cuda:0')
        >>> print(monitor.get_tensor_device(tensor))
        'cuda:0'
        >>> monitor.print_cuda_info()
        CUDA Device Information:
          Current device: 0
          Device name: NVIDIA A100
          ...
    """

    @staticmethod
    def get_tensor_device(tensor: Any) -> str:
        """
        Get the device of a tensor in a human-readable format.

        This method safely determines the device location of a PyTorch tensor,
        handling non-tensor inputs gracefully.

        Args:
            tensor (Any): The tensor whose device is to be identified. Can be
                any type, but only torch.Tensor will return a device location.

        Returns:
            str: The device of the tensor (e.g., 'cuda:0', 'cpu'), or
                'not a tensor' if the input is not a torch.Tensor.

        Example:
            >>> tensor = torch.randn(10, device='cuda:0')
            >>> DeviceMonitor.get_tensor_device(tensor)
            'cuda:0'
            >>> DeviceMonitor.get_tensor_device([1, 2, 3])
            'not a tensor'
        """
        if not isinstance(tensor, torch.Tensor):
            return "not a tensor"
        return str(tensor.device)

    @staticmethod
    def get_current_device() -> str:
        """
        Get the current active CUDA device if available.

        This method determines the currently active CUDA device in the PyTorch
        context, falling back to CPU if CUDA is not available.

        Returns:
            str: The current CUDA device in the format 'cuda:<device_id>' or
                'cpu' if CUDA is not available.

        Example:
            >>> DeviceMonitor.get_current_device()
            'cuda:0'  # If CUDA is available
            >>> DeviceMonitor.get_current_device()
            'cpu'     # If CUDA is not available
        """
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    @staticmethod
    def print_cuda_info() -> None:
        """
        Print detailed information about the current CUDA device and its memory usage.

        This method provides comprehensive information about the CUDA device,
        including device name, total memory, allocated memory, and cached memory.
        It's useful for debugging memory issues and monitoring resource usage.

        The information printed includes:
        - Current device ID
        - Device name
        - Total memory (MB)
        - Currently allocated memory (MB)
        - Cached memory (MB)

        Note:
            - Memory values are reported in megabytes (MB)
            - If CUDA is not available, indicates CPU-only operation
            - Memory statistics are point-in-time snapshots
        """
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_props = torch.cuda.get_device_properties(current_device)
            print(f"\nCUDA Device Information:")
            print(f"  Current device: {current_device}")
            print(f"  Device name: {device_props.name}")
            print(f"  Total memory: {device_props.total_memory / 1024**2:.2f} MB")
            print(
                f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )
            print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        else:
            print("\nNo CUDA device available. Running on CPU.")

    @staticmethod
    def device_trace(func: Callable) -> Callable:
        """
        Decorator for tracing tensor device locations and execution time in functions.

        This decorator provides detailed information about tensor device locations
        before and after function execution, as well as execution timing. It's
        particularly useful for debugging device-related issues and performance
        monitoring.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            Callable: The wrapped function with added device tracing and timing
                capabilities.

        The decorator tracks:
        - Current active device
        - Device location of all tensor arguments
        - Device location of tensor keyword arguments
        - Device location of tensor results
        - Function execution time

        Example:
            >>> @DeviceMonitor.device_trace
            ... def add_tensors(a, b):
            ...     return a + b
            >>> x = torch.randn(10, device='cuda:0')
            >>> y = torch.randn(10, device='cuda:0')
            >>> result = add_tensors(x, y)
            Tracing device locations in add_tensors:
              Current device: cuda:0
              Arg 0 device: cuda:0
              Arg 1 device: cuda:0
              Result device: cuda:0
              Execution time: 0.0001 seconds
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\nTracing device locations in {func.__name__}:")
            print(f"  Current device: {DeviceMonitor.get_current_device()}")

            # Track tensor arguments
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    print(f"  Arg {i} device: {DeviceMonitor.get_tensor_device(arg)}")

            for name, arg in kwargs.items():
                if isinstance(arg, torch.Tensor):
                    print(
                        f"  Kwarg {name} device: {DeviceMonitor.get_tensor_device(arg)}"
                    )

            # Time the function execution
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            # Track result device
            if isinstance(result, torch.Tensor):
                print(f"  Result device: {DeviceMonitor.get_tensor_device(result)}")
            elif isinstance(result, (tuple, list)):
                for i, item in enumerate(result):
                    if isinstance(item, torch.Tensor):
                        print(
                            f"  Result[{i}] device: {DeviceMonitor.get_tensor_device(item)}"
                        )

            print(f"  Execution time: {end_time - start_time:.4f} seconds")
            return result

        return wrapper
