import torch
import time
from functools import wraps


class DeviceMonitor:
    """A utility class for monitoring tensor device locations and operations in PyTorch."""

    @staticmethod
    def get_tensor_device(tensor):
        """Returns the device of a given tensor in a human-readable format.

        Args:
            tensor (torch.Tensor): The tensor whose device is to be identified.

        Returns:
            str: The device of the tensor, or "not a tensor" if the input is not a tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            return "not a tensor"
        return str(tensor.device)

    @staticmethod
    def get_current_device():
        """Returns the current CUDA device if available, otherwise returns 'cpu'.

        Returns:
            str: The current CUDA device in the format 'cuda:<device_id>' or 'cpu' if CUDA is not available.
        """
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    @staticmethod
    def print_cuda_info():
        """Prints detailed information about the current CUDA device if available, otherwise indicates that no CUDA device is available."""
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
    def device_trace(func):
        """A decorator to trace tensor device locations in functions and measure execution time.

        Args:
            func (callable): The function to be decorated.

        Returns:
            callable: The wrapped function with added device tracing and timing.
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
