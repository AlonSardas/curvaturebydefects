# Curvature design by defects

Simulating molecular dynamics using [HOOMD-blue](https://hoomd-blue.readthedocs.io/) 
to see how the distribution of defects on a lattice changes its Gaussian curvature.

<p style="text-align:center">
<img src="./images/sphere-by-traceless-Q-config.png" alt="flat-config" width="200">
<img src="./images/sphere-by-traceless-Q.png" alt="curved-config" width="200">
</p>

## Installation
To run the code, the following packages must be installed:

```
hoomd
gsd
numpy
matplotlib
```

## Plotting trajectory file
Once we have a trajectory saved as .gsd file we can plot its data by the script
```shell
defectsply.py plot --file FILE
```
