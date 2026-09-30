This document was created with AI assistance and only serves as a reference for the package architecture.
# `pyphasediagram` package architecture

This document describes the modules under `src/pyphasediagram`. The diagrams use
[Mermaid](https://mermaid.js.org/) and render directly in GitHub and many Markdown
viewers.

## Module dependencies

Solid arrows are internal imports. Dashed arrows point to third-party packages.
The two package `__init__.py` files form the public import paths shown at the top.

```mermaid
flowchart LR
    package_init["pyphasediagram<br/>__init__.py"]
    diagram["diagram.py<br/>PhaseDiagram"]
    binodal["binodal.py<br/>Binodal, BinodalSection"]
    spinodal["spinodal.py<br/>Spinodal"]
    point["point.py<br/>point/value classes"]
    stepper_init["stepper/__init__.py<br/>exports Stepper"]
    flory["stepper/flory_stepper.py<br/>Stepper"]
    base["stepper/base.py<br/>BaseStepper"]

    package_init -->|exports| diagram
    diagram --> binodal
    diagram --> spinodal
    binodal --> point
    binodal --> stepper_init
    spinodal --> point
    stepper_init -->|exports| flory
    flory --> base

    subgraph third_party[Third-party dependencies]
        numpy[NumPy]
        matplotlib[Matplotlib]
        shapely[Shapely]
        networkx[NetworkX]
        jax[JAX]
        scipy[SciPy]
    end

    diagram -.-> numpy
    diagram -.-> matplotlib
    diagram -.-> shapely
    binodal -.-> numpy
    binodal -.-> matplotlib
    binodal -.-> shapely
    binodal -.-> networkx
    spinodal -.-> numpy
    spinodal -.-> matplotlib
    spinodal -.-> shapely
    spinodal -.-> networkx
    point -.-> numpy
    point -.-> matplotlib
    point -.-> networkx
    flory -.-> numpy
    flory -.-> jax
    flory -.-> scipy
    base -.-> numpy
    base -.-> jax
```

## Classes, attributes, and methods

`+` denotes public API, `-` denotes an internal implementation detail, and `{static}`
or `{abstract}` describes special methods. Attributes marked "after build" are added
by `build()` rather than by the constructor.

```mermaid
classDiagram
    direction LR

    class PhaseDiagram {
        +ndarray chis
        +Spinodal spinodal (after build)
        +Binodal binodal (after build)
        +__init__(chis)
        -_free_energy(phi) ndarray
        +build(delta=1e-3) None
        +get_compositions(phi_m) ndarray
        +plot_phase_counts(ax=None)
        +plot(ax=None)
        +plot_summary()
    }

    class Spinodal {
        +tuple DOMAIN_CORNERS
        +ndarray chis
        +Graph spinodal_graph
        +int node_id
        +list~CriticalPoint~ critical_points
        +list~Polygon~ polygons
        +__init__(chis)
        +build(num_points=10000)
        -_get_p_q(phi, is_calculate_phi2=True)
        +get_spinodal_coords()
        +phi2_from_phi1(phi1)
        +eigenvalues_from_phi(phi1, phi2)
        -_domain_data(phi1_i, phi1_f, num_points=5000)
        -_spinodal_domains()
        -_in_domain(phi1, phi2) bool
        -_clip_to_domain()
        -_connect_branches()
        -_coords_from_subgraph(sg)
        -_third_derivative(phi1, phi2)
        -_find_critical_points()
        -_build_polygons()
        +get_unstable_manifold()
        +plot(**kwargs)
    }

    class Binodal {
        +float SV_BRANCH_THRESHOLD
        +ndarray chis
        +list~CriticalPoint~ critical_points
        +list~BinodalSection~ binodal_sections
        +list~Polygon~ three_phase_polygons
        +list~Polygon~ two_phase_polygons
        -Stepper _stepper
        -list~BinodalInitialPoint~ _bipt_task_list
        -list~BinodalInitialPoint~ _bipt_hist
        +__init__(chis, critical_points=[])
        -_build_section(phi_init, v_init)
        +build(unstable_manifold=None) None
        -_remove_duplicate_sections()
        -_find_branching_points(section)
        -_find_phase_polygons()
        -_find_three_phase_polygons(overlap_threshold=0.98)
        -_find_two_phase_polygons()
        -_remove_unstable_sections(unstable_manifold=None)
        -_free_energy(phi)
        +decomposition_from_composition(phi_means)
        +plot_sections(**kwargs)
        +plot_polygons(**kwargs)
    }

    class BinodalSection {
        +ndarray phis
        +ndarray svs
        +LineString line_a
        +LineString line_b
        -tuple _lines
        -tuple _coords
        +__init__(phis, svs)
        +plot(colora="red", colorb="blue", is_tie_lines=False, **kwargs)
        -_interpolate_on_segment(coords, point) tuple
        -_intersection_params(self_idx, other, other_idx)
        +intersects_with(other)
        +contained_in(other, n_samples=1000) bool
        -_degenerate_points(sv_branch_threshold=1e-2)
        +__len__() int
    }

    class BaseStepper {
        <<abstract>>
        +float MAX_STEPS
        +int CYCLE_MIN_STEPS
        +float CYCLE_INIT_TOL
        -callable _residual_jit
        -callable _jac_fn
        -callable _svd_fn
        -callable _run_jit
        +__init__()
        +residual(phi)* ndarray
        +is_terminate(phi)* ndarray
        -_svd(J, full_matrices=False)
        -_tangent_vec(phi)
        -_projection(phi)
        -_hit_initial_cycle(step, phi, phi_hist)
        -_decode_status(status)
        -_projection_loop(phi)
        -_step_once(...)
        -_loop_cond(carry)
        -_loop_body(...)
        -_run_impl(...)
        +run(phi_init, v_init, delta_0=2e-4, delta_1=1e-3)
    }

    class Stepper {
        +ndarray chis
        +__init__(chis)
        +residual(phi) ndarray
        +is_terminate(phi) ndarray
        +binary_state(chi) float
        +binary_init(which_comp=0) ndarray
    }

    class Point {
        +int idx
        +float phi1
        +float phi2
        +__init__(idx, phi1, phi2)
        +dist(pt)
        +__repr__() str
        +__add__(pt)
        +__sub__(pt)
        +plot(s=10, **kwargs)
        +is_between(pt1, pt2, tol=pi/8) bool
        +is_close_to(pt, tol=1e-3) bool
        +__iter__()
    }

    class SpinodalPoint {
        +__repr__() str
    }

    class CriticalPoint {
        +float dphi1
        +float dphi2
        +__init__(idx, phi1, phi2, dphi1, dphi2)
        +from_points(idx, pt1, pt2) CriticalPoint
        +get_phi_and_v_init()
        +plot(s=10, **kwargs)
        +__repr__() str
    }

    class BinodalPoint {
        +Point pta
        +Point ptb
        +__init__(phi1a, phi2a, phi1b, phi2b)
        +from_points(pt1, pt2, sv) BinodalPoint
        +__repr__() str
        +to_numpy() ndarray
        +plot(s=20, **kwargs)
    }

    class BinodalInitialPoint {
        +ndarray phi_init
        +ndarray v_init
        +__init__(phi_init, v_init)
        +is_similar_to(bipt, dist_tol=1e-3, angle_tol=pi/8) bool
        +plot(**kwargs)
        +__repr__() str
        +__iter__()
    }

    BaseStepper <|-- Stepper
    Point <|-- SpinodalPoint
    Point <|-- CriticalPoint
    BinodalPoint <|-- BinodalInitialPoint

    PhaseDiagram *-- Spinodal : builds
    PhaseDiagram *-- Binodal : builds
    Spinodal *-- SpinodalPoint : graph nodes
    Spinodal *-- CriticalPoint : finds
    Binodal *-- BinodalSection : contains
    Binodal *-- Stepper : traces with
    Binodal o-- CriticalPoint : initialized from
    Binodal *-- BinodalInitialPoint : queues/history
    BinodalPoint *-- Point : phase endpoints
```

## Main build flow

```mermaid
flowchart TD
    start["PhaseDiagram(chis)"] --> build["PhaseDiagram.build()"]
    build --> spinodal_create["Create Spinodal"]
    spinodal_create --> spinodal_build["Spinodal.build()"]
    spinodal_build --> graph["Construct spinodal graph and polygons"]
    graph --> critical["Find CriticalPoint objects"]
    critical --> binodal_create["Create Binodal with critical points"]
    binodal_create --> manifold["Get unstable manifold from Spinodal"]
    manifold --> binodal_build["Binodal.build(unstable_manifold)"]
    binodal_build --> seeds["Seed from binary limits and critical points"]
    seeds --> trace["Stepper.run() traces coexistence sections"]
    trace --> branch["Detect and queue branch points"]
    branch --> clean["Remove unstable and duplicate sections"]
    clean --> polygons["Build two- and three-phase polygons"]
    polygons --> ready["Diagram ready for plotting/decomposition"]
```

## Reading guide

- `diagram.py` is the facade: start with `PhaseDiagram.build()` and its plotting
  methods.
- `spinodal.py` finds instability boundaries and critical points.
- `binodal.py` traces coexistence curves, detects intersections, and builds the
  two- and three-phase regions.
- `stepper/base.py` contains the generic JAX predictor/projector loop;
  `stepper/flory_stepper.py` supplies the Flory-Huggins residual and initialization.
- `point.py` provides the small point and initialization objects shared by the
  spinodal and binodal calculations.

