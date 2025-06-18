<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/0Y1b8Xvb.jpg" alt="Squeezed Cat"></a>
</p>

<h3 align="center">FoPaFF: Fock Pareto Front Finder</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> A tool for finding bounds between different quantum state metrics using multi-objective optimization techniques.
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Possible Issues](#issues)
- [Built Using](#built_using)
- [Authors](#authors)

## 🧐 About <a name = "about"></a>

FoPaFF (Fock Pareto Front Finder) is a tool designed to find the lower virtual interaction fidelity bound based on the expectation value of a specific operator in a truncated Fock space. It leverages GPU acceleration to perform quantum operations and uses the NSGA-II algorithm to find Pareto optimal solutions.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have the following software installed:

- Python 3.8 or higher
- CUDA-compatible GPU (for GPU acceleration)
- Required Python packages (listed in `requirements.txt`)

### Installing

1. Clone the repository:
   ```sh
   git clone https://github.com/kuchar-one/fopaff.git
   cd fopaff
   ```

2. Install the required Python packages:
   ```sh
   python -m venv. venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 🎈 Usage <a name="usage"></a>

To run the optimization, use the following command:

```sh
python fopaff.py --gpu_id 0 -N 30 -u 3 -c 10 -k 100 --pop_size 500 --max_generations 2000 --verbose
```

This command will start the optimization process with the specified parameters. The results, including metrics and animations, will be saved in the `output` directory. Matrices utilized within the optimization can be found in the `cache` directory.

## ❗ Possible Issues <a name = "issues"></a>

On lower-end GPUs, the available VRAM might be insufficient for initializing many parallel workers. In that case, either lower the `num_workers` variable in `src.nsga_ii.optimize_quantum_state_gpu_cpu` or replace `src.nsga_ii` with `src.nsga_ii.lowend`.

## ⛏️ Built Using <a name = "built_using"></a>

- [PyTorch](https://pytorch.org/) - Deep Learning Framework
- [QuTiP](http://qutip.org/) - Quantum Toolbox in Python
- [pymoo](https://pymoo.org/) - Multi-objective Optimization Framework
- [Matplotlib](https://matplotlib.org/) - Plotting Library
- [NumPy](https://numpy.org/) - Numerical Computing Library

## ✍️ Authors <a name = "authors"></a>

- [Vojtěch Kuchař](https://kuchar.one)
