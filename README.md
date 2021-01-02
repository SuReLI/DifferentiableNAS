# DifferentiableNAS
Implementations of existing and new methods in Differentiable Neural Architecture Search in Julia.

This package will implement existing methods such as DARTS and it's successive variants. It will also provide a framework for developing new differentiable neural architecture search methods.

The `src` folder contains the bulk of the package. Currently, in-progress code for implementing the DARTS Super Network architecture and learning algorithm are in `src/DARTSModel.jl` and `src/DARTSTraining.jl`, respectively.

The `test` folder contains unit tests as well as scripts for experiments.

