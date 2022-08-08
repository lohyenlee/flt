# Fast Laplace Transform
## Yen Lee Loh, 2022-8-7

This repository provides a Python implementation of Fast Laplace transform (FLT) algorithm by Yen Lee Loh.
The discrete Laplace transform (DLT) in statistical mechanical terminology is
    $$ Z_m = \sum_n e^{-\beta_m U_n} W_n \quad(m=1,\dots,N;n=1,\dots,N) $$
where 
$U_n$ are input abscissae (energy levels),
$W_n$ are input ordinates (multiplicities or degeneracies),
$\beta_m$ are output abscissae (inverse temperatures),
and $Z_m$ are output ordinates (values of the partition function).
Brute-force evaluation of the DLT takes $O(MN)$ effort.
The FLT algorithm usually takes between $O(M+N)$ and $O( (M+N)^{1.5} )$ effort.
It computes the DLT to a given fractional error even when the inputs vary over very large ranges.
`flt.ipynb` is a JupyterLab notebook (tested on conda 4.12.0 on Ubuntu 20.04 with JupyterLab 3.3.2).
