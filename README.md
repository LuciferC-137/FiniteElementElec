# Finite Element Method Introduction

## Introduction

This repo is a collection of notes and codes for the Finite Element Method (FEM). This code is a simple 2D FEM in a simple square mesh intended to demonstrate the peak effect in electric fields. The code is written in Python.

**What you can do with this package**

The package allows you to define a mesh, fix some boundary conditions and solve the potential field according to Poisson equation for the rest of the mesh.

There are two geometries available, but you can also define your own, as long as you respect some prerequisites. Those two default
geometries are a circle and a square. The circle is defined by its radius and an angle for the peak, and the square is defined by its side length.

## Project Structure

```
fem_peak25/
├── docs/
│   ├── img/
│   ├── biblio.bib
│   ├── main.pdf
│   └── main.tex
├── output/
├── examples/
│   ├── peak_circle.py
│   └── peak_square.py
├── fem_peak25/
│   ├── __init__.py
│   ├── elements.py
│   ├── logger.py
│   ├── solver.py
│   └── plotting.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

The main package functions are in `fem_peak25/` and the examples lies in `examples/`. Those example
files do not need to install the package to be ran. The documentation is in `docs/` (already compiled).

## Installation

**NOTE** Installation is not necessary to run the examples. You can run the examples directly from the root of the project see the section Quick Start below.

To install the package, navigate to the root of the project (the folder where the `pyproject.toml` is) and run the following command:

```bash
pip install .
```

You can now use the classes of the package in any of your codes by importing the class as such:

```python
from fem_peak25.elements import Mesh, MeshBuilder
from fem_peak25.solver import Solver
from fem_peak25.plotting import Plotter
```

## Quick Start

If you simply need to test the package, you can run the examples in the `examples/` folder. To run the examples, navigate to the root of the project and run one of the following command:

```bash
python examples/peak_circle.py
```

```bash
python examples/peak_square.py
```

## Documentation

The documentation is in the `docs/` folder. The main file is `main.tex`. To compile the documentation, you need to have a LaTeX compiler installed. The documentation is
compiled by default in the `docs/` folder, therefore you can also access the compiled documentation in the `docs/main.pdf` file.

This documentation is written in french and describes the mathematical background of the Finite Element Method and the implementation of the package.

## Define your own geometry

To define your own geometry, you need to create a class that inherits from the `Mesh` class and implement the `build` method. This method should return a `Mesh` object. You can see the `CircleMesh` and `SquareMesh` classes in the `fem_peak25/elements.py` file for examples.

If you define your own geometry, you can not use the `MeshBuilder` class to facilitate the construction of your mesh, all should be done by hand. You will need to:
- Make a dictionnary of nodes
- Index all the neighbors of the nodes
- Create all elements as a combination of exactly 3 nodes.

Once this is done, you can define some values
for some nodes that will be treated as boundary conditions, then you can pass your mesh to the solver.

Every mesh will always have a "peak" node, which is the center of the mesh. It is usually used to compute the "angle from center" to define a proper potential boundary conditions that you can see in the example scripts. But nothing bounds you to use this peak node as a peak, you can use it as any other node.