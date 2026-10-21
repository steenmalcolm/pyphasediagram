---
title: 'pyphasediagram: A Python package for tracing phase coexistence in ternary mixtures'
tags:
  - Python
  - Soft Matter
  - Phase Coexistence
  - Ternary Mixtures
authors:
  - name: Malcolm Steen
    orcid: xxx
    affiliation: "1,2"
  - name: Stefan Köstler
    orcid: xxx
    affiliation: "1,2"
  - name: David Zwicker
    orcid: xxx
    affiliation: 1
affiliations:
  - index: 1
    name: 
  - index: 2
    name: 
date: 21 October 2026
bibliography: paper.bib
---

The exact requirements for this document can be found [here](https://joss.readthedocs.io/en/latest/paper.html)

# Summary

- mixtures can separate into phases with different compositions, as in oil-water demixing.
- according to Gibbs law two or three phases can coexist in ternary sytstems.
- `pyphasediagram` constructs and visualizes ternary diagrams for incompressible Flory–Huggins mixtures.
- inputs: three pairwise interaction parameters.
- outputs: coexistence curves, instability separatrix, critical points and two- and three-phase regions.
- numerical continuation plus geometric analysis
- coexistence curves represented as explicit objects.

# Statement of need

- Why three components: simple binary Flory–Huggins models have one independent composition and can be reduced to implicit coexistence relations. Ternary mixtures have two independent composition fields, making them a minimal setting for additional multicomponent effects.
- These effects include:
  - enrichment of the third species at the interface, which affects the surface tension [@Huang1995].
  - the existence of three distinct phases and triangular coexistence regions [@Huang1995].
  - Rich topological features in phase diagram
  - etc.

- the coexistence states for neighboring interaction parameter or compositions can inform about the dynamics of the system (Stefans paper).
- repeated `flory` equilibrium calculations are expensive for this purpose.
- organize branches, discover critical points, and identify three-phase regions.
- target users: researchers studying multicomponent phase separation in soft matter and liquid mixtures.
- what is the contribution? Complete tracing and organization of coexistence geometry (two- and three-phase regions).

# State of the field

- Polymer binodal calculations: @Wei2006 examined binodal predictions and estimated the Flory–Huggins interaction parameters.
- Genetic optimization: @Milani2007 minimized an objective based on chemical-potential differences between phases. This establishes an algorithmic approach to computing coexing phases
- Convexification (closest numerical match): @Mao2019 constructed convex hulls of discretized free-energy landscapes. Hull resolution depends on sampling rate (perhaps mention scaling relation for resolution which I believe is quadratic vs linear for this algorithm)
- Closest theoretic match: @Shcherbakova2023 represented binodals in ternary systems as curves in the four-dimensional space of paired compositions and traces them. 
- predictor–corrector continuation with Newton correction [@Allgower1990]
- Package-specific features: singular-value based prediction and correction, multiple initialization strategies, branch exploration, and phase-region construction in composition space.
- `flory`: free-energy minimization over multicomponent systems.
  Good for phase-diagram sampling, particularly in high component systems [@Qiang2025].
- vs `pyphasediagram` represents complete ternary coexistence curves and their (nearly) exact **geometry**
- `pycalphad` [@Otis2017] and `Equilipy` [@Kwon2024] provide database-driven methodology.
- Rationale for a separate package: tracing, branch bookkeeping, and composition space geometry.
- Trade-off: a high resolution diagram construction with less generality than existing equilibrium solvers.

# Software design

- Free-energy model: three volume fractions with incompressibility $\phi_0+\phi_1+\phi_2=1$.
- dimensionless free-energy density:

  $$
  f=\sum_{i=0}^{2}\phi_i\ln\phi_i
    +\sum_{0\leq i<j\leq2}\chi_{ij}\phi_i\phi_j.
  $$

- coexistence constraints: equal exchange chemical potentials $\mu_i$ and osmotic pressure $\Pi$:

  $$
  \mathbf{E}(\mathbf{x})=
  \begin{pmatrix}
  \mu_1^{(a)}-\mu_1^{(b)}\\
  \mu_2^{(a)}-\mu_2^{(b)}\\
  \Pi^{(a)}-\Pi^{(b)}
  \end{pmatrix}=\mathbf{0}.
  $$

- predictor: step along the tangent to obtain a neighboring approximate state.
- corrector: move back towards coexistence manifold through projection.
- initialization: binary limits and critical points from spinodal analysis
- branch discovery: local minima of the smallest finite singular value allows discovery of every branch.
- JAX: automatic differentiation allows flexibility and ease when adopting new coexistence conditions and provides significant speed-up for tracing, which is the bottle neck of the algorithm [@Bradbury2018]. 
- Postprocessing: remove locally unstable and duplicate sections. Assemble three-phase triangles from intersections.
- Geometry: graph and polygon representations support connectivity and intersection operations in order to get a minimal and complete representation of the phase diagram.

# Research impact statement

# AI usage disclosure

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements


# References
