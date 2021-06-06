# vdist-solver
Velocity distribution function solver using Liouville's theorem

|vx-vz distribution|vz distribution|
|---|---|
|![16_16_65](https://user-images.githubusercontent.com/71783375/120922088-2df7ab00-c702-11eb-876b-1b4538c2c9ad.png)|![vdist_ion_z](https://user-images.githubusercontent.com/71783375/120922107-3fd94e00-c702-11eb-901a-32576eccf53b.png)|

## Requirement
* numpy
* matplotlib
* tqdm
* scipy
* emout

## Installation
```
pip install git+https://github.com/Nkzono99/vdist-solver.git
```

## Usage
```
> gen-vsolver 'xz'  # Create template script of vdist-solver

> vim vdist-solver.py  # Change settings

> python vdist-solver.py --help  # show help message

> python vdist-solver.py -d "<directory>"  # Plot distribution
```
