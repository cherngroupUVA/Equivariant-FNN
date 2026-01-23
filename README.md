# Repository Structure and Data Description

This repository implements a machine-learning (ML) framework based on the locality principle, in which irreducible representations (IRs) decomposed from the input features are used as the fundamental building blocks of an equivariant fully connected neural network (E-FNN).

-------------------------------------------------
Training Data
-------------------------------------------------
The training_data zip file contains four .pt files:

Q_random_200.pt: 200 random initial lattice configurations

force_random_200.pt: Force snapshots corresponding to Q_random_200.pt

Q_random_400.pt: 400 lattice snapshots sampled from the post-quench dynamical evolution

force_quench_400.pt: Force snapshots corresponding to Q_random_400.pt

All data are defined on a 40 × 40 square lattice.

-------------------------------------------------
Code
-------------------------------------------------

The code directory contains two subfolders:

train/ — Training the ML Model

This folder includes scripts for training the equivariant FNN:

training_script.py: Main training script for the equivariant FNN

model.py: Defines the architecture of the ML model

holstein_generate_IRS.py: Generates irreducible representations (IRs) from neighbor representations

generate_feature.py: Includes the function that decomposes input features into IR components

read_neighbor.py: Reads the neighbor information for each central lattice site

dynamic/ — Dynamical Simulations

This folder contains scripts required for dynamical simulations:

simulation_200.py: Main script for running dynamical simulations on a 200 × 200 lattice

main_200.py: Specifies the initial configurations and simulation parameters

model.py: Defines the structure of the ML model used in dynamics

dynamics_200.py: Implements the Holstein model and its equations of motion


-------------------------------------------------
Neighbor Information
-------------------------------------------------

The neighbor directory contains CSV files specifying the neighbor indices for each lattice site. These files are used to construct local environments for the machine-learning model.
