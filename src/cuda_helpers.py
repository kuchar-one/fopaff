import torch
import time
from functools import wraps

class DeviceMonitor:
    """
    Utility class for monitoring and managing PyTorch tensor device locations and operations.
    
    This class provides static methods for tracking tensor device locations, monitoring
    CUDA device usage, and tracing tensor operations across devices. It is particularly
    useful for debugging device-related issues in GPU-accelerated applications.

    Methods:
        get_tensor_device: Get the device location of a tensor.
        get_current_device: Get the current active CUDA device.
        print_cuda_info: Print detailed information about available CUDA devices.
        device_trace: Decorator for tracing tensor device locations in functions.
    """
    
    @staticmethod
    def get_tensor_device(tensor: torch.Tensor) -> str:
        """
        Get the device of a tensor in a human-readable format.
        
        Args:
            tensor (torch.Tensor): The tensor to check the device of.
        
        Returns:
            str: The device of the tensor (e.g., 'cuda:0', 'cpu') or
                'not a tensor' if the input is not a tensor.
            
        Note:
            This method is useful for debugging device-related issues and
            ensuring tensors are on the expected devices.
        """
        if not isinstance(tensor, torch.Tensor):
            return "not a tensor"
        return str(tensor.device)
    
    @staticmethod
    def get_current_device() -> str:
        """
        Get the current active CUDA device if available.
        
        Returns:
            str: The current CUDA device (e.g., 'cuda:0') or 'cpu' if CUDA
                is not available.
                
        Note:
            This method is useful for verifying the active device before
            performing tensor operations.
        """
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    
    @staticmethod
    def print_cuda_info() -> None:
        """
        Print detailed information about the current CUDA device.
        
        This method provides comprehensive information about the CUDA device,
        including device name, total memory, allocated memory, and cached memory.
        
        Note:
            - If CUDA is available, prints:
              * Current device ID
              * Device name
              * Total memory (MB)
              * Allocated memory (MB)
              * Cached memory (MB)
            - If CUDA is not available, indicates CPU-only operation
        """
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            device_props = torch.cuda.get_device_properties(current_device)
            print(f"\nCUDA Device Information:")
            print(f"  Current device: {current_device}")
            print(f"  Device name: {device_props.name}")
            print(f"  Total memory: {device_props.total_memory / 1024**2:.2f} MB")
            print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        else:
            print("\nNo CUDA device available. Running on CPU.")
    
    @staticmethod
    def device_trace(func):
        """
        Decorator to trace tensor device locations and execution time in functions.
        
        This decorator provides detailed information about tensor device locations
        before and after function execution, as well as execution timing. It's
        particularly useful for debugging device-related issues and performance
        monitoring.

        Args:
            func (callable): The function to be decorated.
        
        Returns:
            callable: The wrapped function with device tracing capabilities.
            
        Note:
            The decorator tracks:
            - Current active device
            - Device location of all tensor arguments
            - Device location of tensor keyword arguments
            - Device location of tensor results
            - Function execution time
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
                    print(f"  Kwarg {name} device: {DeviceMonitor.get_tensor_device(arg)}")
            
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
                        print(f"  Result[{i}] device: {DeviceMonitor.get_tensor_device(item)}")
            
            print(f"  Execution time: {end_time - start_time:.4f} seconds")
            return result
        return wrapper


def set_gpu_device(device_id: int) -> None:
    """
    Set the active GPU device for computation.

    This function sets the specified GPU device as the active device for PyTorch
    operations. It includes validation checks to ensure the requested device is
    available.

    Args:
        device_id (int): Integer ID of the GPU device to use.
    
    Raises:
        ValueError: If the specified device_id is not available.
        RuntimeError: If no CUDA-capable GPU devices are found.
        
    Note:
        - Prints confirmation message with device name if successful
        - Checks CUDA availability before attempting to set device
    """
    if torch.cuda.is_available():
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"GPU device {device_id} not found. Available devices: 0-{torch.cuda.device_count()-1}")
        torch.cuda.set_device(device_id)
        print(f"Using GPU device {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        raise RuntimeError("No CUDA-capable GPU devices found")


class GPUDeviceWrapper:
    """
    Wrapper class that ensures tensor operations are performed on a specific GPU device.
    
    This class wraps an instance of another class and ensures that all tensor
    operations performed by that instance are executed on the specified GPU device.
    It automatically handles device movement for tensor inputs and outputs.

    Args:
        wrapped_instance (object): The instance to wrap. Should be a class that
            performs tensor operations.
        device_id (int): The ID of the GPU device to use for all tensor operations.
        
    Attributes:
        device (str): The CUDA device identifier (e.g., 'cuda:0').
        _wrapped (object): The wrapped instance.
        
    Note:
        - Automatically moves input tensors to the specified device
        - Ensures output tensors are on the specified device
        - Handles nested tensor structures (lists, tuples)
        - Preserves non-tensor attributes and methods of the wrapped instance
    """
    def __init__(self, wrapped_instance, device_id: int):
        self._wrapped = wrapped_instance
        self.device = f'cuda:{device_id}'
        
        # Force the base class's tensors to the correct device
        self._wrapped.device = self.device
        self._wrapped.d = self._wrapped.d.to(self.device)
        self._wrapped.identity = self._wrapped.identity.to(self.device)
        
    def __getattr__(self, name: str):
        """
        Redirect attribute access to the wrapped instance with device management.
        
        This method intercepts attribute access to the wrapped instance and ensures
        that any tensor operations are performed on the correct device. It handles
        both method calls and attribute access.

        Args:
            name (str): The name of the attribute to access.
        
        Returns:
            The attribute from the wrapped instance, with tensor operations
            redirected to the correct device.
            
        Note:
            - For method calls: moves input tensors to the correct device and
              ensures outputs are on the correct device
            - For tensor attributes: moves them to the correct device
            - For non-tensor attributes: returns them as-is
        """
        attr = getattr(self._wrapped, name)
        if callable(attr):
            def wrapped_method(*args, **kwargs):
                new_args = [arg.to(self.device) if torch.is_tensor(arg) else arg for arg in args]
                new_kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
                result = attr(*new_args, **new_kwargs)
                if torch.is_tensor(result):
                    return result.to(self.device)
                elif isinstance(result, (list, tuple)):
                    return type(result)(x.to(self.device) if torch.is_tensor(x) else x for x in result)
                return result
            return wrapped_method
        elif torch.is_tensor(attr):
            return attr.to(self.device)
        return attr