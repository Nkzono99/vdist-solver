# vdist-solver
Velocity distribution function solver using Liouville's theorem

|vx-vz distribution|vz distribution|backtrace|
|---|---|---|
|![16_16_65](https://user-images.githubusercontent.com/71783375/120922088-2df7ab00-c702-11eb-876b-1b4538c2c9ad.png)|![vdist_ion_z](https://user-images.githubusercontent.com/71783375/120922107-3fd94e00-c702-11eb-901a-32576eccf53b.png)|![backtrace_test](https://user-images.githubusercontent.com/71783375/120922700-a0b65580-c705-11eb-9410-85b841e5718d.png)|

## Requirement
* numpy
* matplotlib
* tqdm
* scipy
* emout
* mpi4py (Install by yourself if use MPI)

## Installation
```
pip install vdist-solver
```

## Simple Usage for EMSES
``` 
# Use velocity distribution function solver
> gen-vdsolver 'xz'  # Create template script of vdist-solver
> vim vdist-solver.py  # Change settings
> python vdist-solver.py -d "<directory>"  # Plot distribution

# Use backtrace solver
> gen-backtrace 'xz'  # Create template script of backtrace-solver
> vim backtrace-solver.py  # Change settings
> python backtrace-solver.py -d "<directory>"  # Plot backtrace-orbit of particle
```

A detailed description is provided below 
- [Get Started (Japanese)](/readme/GetStarted_Japanese.md).
- [Theory (Japanese)](/readme/Theory_Japanese.md)
  