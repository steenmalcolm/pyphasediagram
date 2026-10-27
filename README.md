# Phase Diagram Analysis for Ternary Flory–Huggins Systems

This project provides tools to **compute and analyze ternary phase diagrams** based on the **Flory–Huggins free energy model**.  
It includes a core class:

- **`PhaseDiagram`** – add binodal branches to a full ternary phase diagram and locate three-phase coexistence points and critical points

## Installation

After cloning the repository and navigating into the project directory, install the package using pip:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

There is a simple example in the `examples` folder.

## Development

Install the pre-commit hooks to ensure code formatting using 

```bash
pip install pre-commit
pre-commit install
```