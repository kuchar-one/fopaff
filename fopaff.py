"""
Quantum State Optimization Command Line Interface

This script provides a command-line interface for running quantum state
optimization using the NSGA-II algorithm. It supports optimization of both
even and odd parity states with customizable parameters.

The script creates necessary directories for caching operators and storing
output results, then processes command line arguments to configure and run
the optimization.

Example:
    Basic usage with default parameters:
    $ python fopaff.py

    Custom optimization with specific parameters:
    $ python fopaff.py --gpu_id 0 -N 30 -u 3 -c 10 -k 100 --pop_size 500 \\
                       --max_generations 2000 --verbose --parity even

Note:
    - Creates 'cache' and 'output' directories if they don't exist
    - Output directory structure includes optimization parameters in the name
    - Supports both even and odd parity quantum states
    - Uses GPU acceleration for quantum operations
"""

import os
import sys
import argparse
from src.helpers import run

if not os.path.exists("cache"):
    os.makedirs("cache")
    os.makedirs("cache/operators")

if not os.path.exists("output"):
    os.makedirs("output")


def main():
    """
    Main function for parsing command line arguments and running optimization.

    This function sets up the argument parser with detailed help messages,
    processes the command line arguments, creates necessary output directories,
    and initiates the quantum state optimization process.

    The following parameters can be configured:
    - GPU device ID
    - Fock space dimension (N)
    - Quantum operation parameters (u, c, k)
    - Population size and maximum generations for NSGA-II
    - Convergence tolerance
    - State parity (even/odd)
    - Verbosity level

    Note:
        - Creates output directory with parameters encoded in the name
        - Supports both even and odd parity optimizations
        - All parameters have reasonable defaults for typical use cases
    """
    parser = argparse.ArgumentParser(
        description="Quantum State Optimization Script",
        epilog="Run optimization with customizable parameters",
    )

    # Add arguments with type, default, and help text
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID to use for computation (default: 0)",
    )
    parser.add_argument(
        "-N", type=int, default=30, help="Fock space dimension N (default: 30)"
    )
    parser.add_argument("-u", type=float, default=3, help="Parameter u (default: 3)")
    parser.add_argument("-c", type=float, default=10, help="Parameter c (default: 10)")
    parser.add_argument("-k", type=int, default=100, help="Parameter k (default: 100)")
    parser.add_argument(
        "--pop_size",
        type=int,
        default=500,
        help="Population size for optimization (default: 500)",
    )
    parser.add_argument(
        "--max_generations",
        type=int,
        default=2000,
        help="Maximum number of generations (default: 2000)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-4,
        help="Tolerance for convergence (default: 5e-4)"
    )
    parser.add_argument(
        "--parity",
        type=str,
        default="even",
        help="Parity of the state (default: even)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call run function with parsed arguments
    if len(sys.argv) > 1:
        if args.parity == "even":
            out_dir = f"output/{args.max_generations}_maxgens_{args.pop_size}_individuals_N{args.N}_u{args.u}_c{args.c}_k{args.k}"
        else:
            out_dir = f"output/{args.max_generations}_maxgens_{args.pop_size}_individuals_N{args.N}_u{args.u}_c{args.c}_k{args.k}_odd"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        run(
            gpu_id=args.gpu_id,
            N=args.N,
            u=args.u,
            c=args.c,
            k=args.k,
            pop_size=args.pop_size,
            max_generations=args.max_generations,
            verbose=args.verbose,
            tolerance=args.tolerance,
            parity=args.parity,
        )


if __name__ == "__main__":
    main()
