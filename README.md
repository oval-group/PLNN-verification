# Piecewise Linear Neural Networks Verification: A comparative study

This repository contains all the code necessary to replicate the findings
described in the paper: [Piecewise Linear Neural Networks Verification: A
comparative study](https://arxiv.org/abs/1711.00455). If you use it in your research, please cite:

```
@Article{Bunel2017,
  author        = {Bunel, Rudy and Turkaslan, Ilker and Torr, Philip H.S and Kohli, Pushmeet and Kumar, M Pawan},
  title        =  {Piecewise Linear Neural Networks Verification: A comparative study},
  journal      = {arxiv:1711.00455},
  year         = {2017},
}
```

The methods contained in this repository are:
* Neural Network verification as a Mixed Integer Program feasibility problem
* Neural Network verification as a Global Optimization problem, solved through
  Branch and Bound

In addition, this also contains conversion scripts to operate other solvers,
included as submodules. If you make use of them, please cite the corresponding paper.
* Reluplex, in `./ReluplexCav2017`
* Planet, in `./Planet`


## Structure of the repository
* `./convex_adversarial` is a git submodule, linking to [the Provably Robust
  Neural Network repository](https://github.com/locuslab/convex_adversarial)
* `./planet/` is a git submodule, linking
  to [the official Planet repository](https://github.com/progirep/planet)
* `./ReluplexCav2017/` is a git submodule, linking to a fork
of
[the official Reluplex repository](https://github.com/guykatzz/ReluplexCav2017).
The fork was made to include some additional code to support additional
experiments that the originally included ones.
* `./plnn/` contains the code for the MIP solver and the BaB solver.
* `./tools/` is a set of python tools used to go from one solver's format to
  another, run a solver on some property, compare experimental results, or
  generate datasets.
* `./scripts/` is a set of bash scripts, instrumenting the tools of `./tools` to
  reproduce the results of the paper.
  
## Running the code
### Dependencies
The code was implemented assuming to be run under `python3.6`.
We have a dependency on:
* [The Gurobi solver](http://www.gurobi.com/) to solve the LP arising from the
Network linear approximation and the Integer programs for the MIP formulation.
Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
* [Pytorch](http://pytorch.org/) to represent the Neural networks and to use as
  a Tensor library. 
* The python packages `psutil` to measure memory usage in our benchmarks and
`sh` to instrument other solvers. 
* **Reluplex** and **Planet** have their own dependency, described in their
  Readme page.

  
### Installing everything
We recommend installing everything into a python virtual environment.

```bash
git clone --recursive https://github.com/oval-group/PLNN-verification.git

cd PLNN-verification
virtualenv -p python3.6 ./venv
./venv/bin/activate

# Install gurobipy to this virtualenv
# (assuming your gurobi install is in /opt/gurobi701/linux64)
cd /opt/gurobi701/linux64/
python setup.py install
cd -

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 

# Install psutil
pip install psutil

# Install the code of this repository
python setup.py install

# Additionally, install the code for Planet and Reluplex
# by cd-ing into their directory and following their 
# installation instructions.
## Reluplex:
cd ReluplexCav2017/glpk-4.60
./configure_glpk.sh
make
make install
cd ../reluplex
make
cd ../check_properties
make
cd ../..

## Planet
cd planet/src
# sudo apt install valgrind qt5-qmake libglpk-dev # if necessary 
qmake
make
# if you encounter linker issues, move -lsuitesparseconfig to the end of the flag list
cd ../..

## Install the code for computing fast heuristic bounds
cd convex_adversarial
python setup.py install
```

### Running the experiments
If you have setup everything according to the previous instructions, you should
be able to replicate the experiments of the paper. To do so, follow the
following instructions:

```bash
## Generate the datasets
# Generate the .rlv (planet/BaB/MIP inputs file from the Acas .nnet files)
./scripts/convertACAS2rlv.sh

# Generate the .nnet files (reluplex inputs) from the CollisionDetection .rlv files
./scripts/convertrlv2rlpx.sh

# Generate the .rlv and .nnet files for the TwinStream dataset
./scripts/generate_twin_ladder_benchmarks.sh

## Generate the results
./scripts/bab_runscript.sh
./scripts/mip_runscript.sh
./scripts/planet_runscript.sh
./scripts/reluplex_runscript.sh

## Analyse the results
# (might have to `pip install matplotlib` to generate curves)
./scripts/generate_analysis_images.sh
# ACAS comparison
./tools/compare_benchmarks.py results/ACAS/reluplex/ results/ACAS/planet/ results/ACAS/MIP/ results/ACAS/BaB/
# collisionDetection comparison
./tools/compare_benchmarks.py results/collisionDetection/reluplex/ results/collisionDetection/planet/ results/collisionDetection/MIP/ results/collisionDetection/BaB/
# TwinStream comparison
./tools/compare_benchmarks.py results/twinLadder/reluplex/ results/twinLadder/planet/ results/twinLadder/MIP/ results/twinLadder/BaB --all_unsat
# Comparison of Linear Approximation quality
./scripts/linear_approximation_comparison.sh

```
