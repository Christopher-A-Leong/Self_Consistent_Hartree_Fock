# Self-Consistent Hartree Fock

Code package designed for computing a self-consistent Hartree Fock computation on Euclidean (honeycomb, square, and Bernal bilayer honeycomb) and hyperbolic lattices with nearest-neighbor Coulomb or onsite Hubbard interactions.

## Overview

Interacting many-body physics problems are difficult to approach computationally due to the curse of dimensionality, whereby the dimensions of the Hamiltonian quickly expands with system size. One solution to this problem is to treat the system within the mean-field formulation, thereby decomposing interacting terms at the cost of quantities which must be solved self-consistently. This latter method is implemented here, on a few preprogrammed lattice geometries; namely a honeycomb flake, a square lattice, and a Bernal bilayer honeycomb lattice (formed by Bernal stacking two honeycomb flakes), as well as hyperbolic lattices, represented on the Poincare disk, with coordination number q=3 or q=4. Upon receiving an initial guess for self-consistently obtained values, the algorithm successively performs exact diagonalizations to obtain new values of these quantities. The algorithm completes once a specified convergence tolerance is reached.

## Requirements

See environment.yml file in the repository to create the conda environment related to this project.
