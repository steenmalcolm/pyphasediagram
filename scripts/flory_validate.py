"""
Validate pyphasediagram binodal computations against the flory package.

Samples random chi matrices, builds phase diagrams, and checks that
tie-line compositions (two-phase) and three-phase regions agree with
flory.find_coexisting_phases.  Results are written to a JSON file.
"""

import argparse
import json
import signal
from pathlib import Path

import flory
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely
from tqdm import tqdm

from pyphasediagram.binodal import Binodal
from pyphasediagram.spinodal import Spinodal


# ---------------------------------------------------------------------------
# Phase-region area computation
# ---------------------------------------------------------------------------
def _compute_phase_areas(binodal: Binodal):
    """Return (two_phase_area, three_phase_area) for sample-count allocation."""
    area_2phase = shapely.unary_union(binodal.two_phase_polygons).area
    area_3phase = shapely.unary_union(binodal.three_phase_polygons).area
    return area_2phase, area_3phase


# ---------------------------------------------------------------------------
# Timeout helper (SIGALRM on Unix, skipped on Windows)
# ---------------------------------------------------------------------------
_HAS_SIGALRM = hasattr(signal, "SIGALRM")


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


def _set_alarm(seconds: int):
    if _HAS_SIGALRM:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)


def _cancel_alarm():
    if _HAS_SIGALRM:
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Chi-matrix generation (mirrors diagram.py main loop)
# ---------------------------------------------------------------------------
def random_chi_matrices(
    rng: np.random.Generator, chi: float = 2.75, spread: float = 0.4
):
    chi_01, chi_02, chi_12 = (rng.random(3) - 0.5) * spread + chi

    chis_3x3 = np.array(
        [
            [0, chi_01, chi_02],
            [chi_01, 0, chi_12],
            [chi_02, chi_12, 0],
        ]
    )

    chi_11 = -2 * chi_01
    chi_22 = -2 * chi_02
    chi_12_r = chi_12 - chi_01 - chi_02
    chis_reduced = np.array([[chi_11, chi_12_r], [chi_12_r, chi_22]])

    return chis_3x3, chis_reduced


# ---------------------------------------------------------------------------
# Flory-Huggins free energy
# ---------------------------------------------------------------------------
def flory_energy(chis: np.ndarray, phi: np.ndarray) -> float:
    """Flory-Huggins free energy of mixing in the reduced two-component form.

    Parameters
    ----------
    chis : (2, 2) array
        Reduced chi-parameter matrix.
    phi : (2,) array
        Volume fractions of components 1 and 2 (component 0 is 1 - phi[0] - phi[1]).

    Returns
    -------
    float
        f = sum_i phi_i ln(phi_i) + 0.5 * phi^T chi phi
        where phi = (phi_1, phi_2) and the solvent fraction is implicit.
    """
    phi = np.asarray(phi, dtype=float)
    phi0 = 1.0 - phi[0] - phi[1]
    # Entropic term over all three components
    all_phi = np.array([phi0, phi[0], phi[1]])
    entropic = np.sum(all_phi * np.log(all_phi))
    # Enthalpic term in reduced form
    enthalpic = 0.5 * phi @ chis @ phi
    return float(entropic + enthalpic)


# ---------------------------------------------------------------------------
# Two-phase validation
# ---------------------------------------------------------------------------
def validate_two_phase(
    binodal: Binodal,
    chis_3x3: np.ndarray,
    chis_reduced: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 10,
    tol: float = 5e-2,
) -> list[dict]:
    """Sample tie-line points and compare to flory.  Returns per-sample records."""

    # Collect non-degenerate tie lines outside three-phase regions
    all_tie_lines = []
    for section in binodal.binodal_sections:
        n_pts = section.phis.shape[2]
        margin = max(1, n_pts // 20)
        for idx in range(margin, n_pts - margin):
            phi_a = section.phis[0, :, idx]
            phi_b = section.phis[1, :, idx]
            if np.linalg.norm(phi_a - phi_b) < 0.05:
                continue
            mid = 0.5 * (phi_a + phi_b)
            mid_pt = shapely.Point(mid)
            in_3phase = any(p.contains(mid_pt) for p in binodal.three_phase_polygons)
            if not in_3phase:
                all_tie_lines.append((section, idx))

    if not all_tie_lines:
        return []

    sample_indices = rng.choice(
        len(all_tie_lines),
        size=min(n_samples, len(all_tie_lines)),
        replace=False,
    )

    records = []
    for sample_idx in sample_indices:
        section, tie_idx = all_tie_lines[sample_idx]
        phi_a = section.phis[0, :, tie_idx]
        phi_b = section.phis[1, :, tie_idx]

        # Resample until the point is outside all three-phase regions
        for _ in range(100):
            t = rng.uniform(0.2, 0.8)
            phi_ab = t * phi_a + (1 - t) * phi_b
            pt = shapely.Point(phi_ab)
            if not any(p.contains(pt) for p in binodal.three_phase_polygons):
                break
        else:
            continue

        phi_means = [1 - phi_ab[0] - phi_ab[1], phi_ab[0], phi_ab[1]]

        phases = flory.find_coexisting_phases(3, chis_3x3, phi_means, progress=False)
        flory_phis = phases.fractions[:, 1:]
        n_phases = phases.fractions.shape[0]

        rec = {
            "phi_mean": phi_ab.tolist(),
            "phi_a": phi_a.tolist(),
            "phi_b": phi_b.tolist(),
            "flory_n_phases": n_phases,
        }

        if n_phases == 2:
            expected = np.array([phi_a, phi_b])
            err1 = float(np.max(np.abs(expected - flory_phis)))
            err2 = float(np.max(np.abs(expected - flory_phis[::-1])))
            error = min(err1, err2)
            rec["error"] = error
            rec["pass"] = error < tol
        else:
            rec["error"] = None
            rec["pass"] = False

        if not rec["pass"]:
            f_binodal = t * flory_energy(chis_reduced, phi_a) + (1 - t) * flory_energy(
                chis_reduced, phi_b
            )
            f_flory = float(
                np.sum(
                    [
                        v * flory_energy(chis_reduced, phases.fractions[i, 1:])
                        for i, v in enumerate(phases.volumes)
                    ]
                )
            )
            delta_f = f_binodal - f_flory
            # Flory finds lower energy => error in algorithm
            if delta_f > 0:
                rec["flory_phis"] = flory_phis.tolist()
                rec["flory_volumes"] = phases.volumes.tolist()
                rec["f_binodal"] = f_binodal
                rec["f_flory"] = f_flory
                rec["delta_f"] = f_binodal - f_flory

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Three-phase validation
# ---------------------------------------------------------------------------
def validate_three_phase(
    binodal: Binodal,
    chis_3x3: np.ndarray,
    chis_reduced: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 10,
) -> list[dict]:
    """Sample inside three-phase polygons and check flory returns 3 phases.

    *n_samples* is the total budget, distributed among polygons proportionally
    to their areas (each polygon gets at least 1 sample).
    """
    records = []
    if not binodal.three_phase_polygons:
        return records

    # Distribute samples among polygons by area
    areas = np.array([p.area for p in binodal.three_phase_polygons])
    total_area = areas.sum()
    if total_area == 0:
        return records
    raw = areas / total_area * n_samples
    counts = np.maximum(1, np.round(raw)).astype(int)

    for poly_idx, poly in enumerate(binodal.three_phase_polygons):
        minx, miny, maxx, maxy = poly.bounds

        for _ in range(counts[poly_idx]):
            # Rejection-sample a point inside the polygon
            for _ in range(1000):
                px = rng.uniform(minx, maxx)
                py = rng.uniform(miny, maxy)
                if poly.contains(shapely.Point(px, py)):
                    break
            else:
                continue

            phi_means = [1 - px - py, px, py]
            phases = flory.find_coexisting_phases(
                3, chis_3x3, phi_means, progress=False
            )
            sig = phases.volumes > 0.01
            n_phases = int(sig.sum())

            records.append(
                {
                    "polygon_idx": poly_idx,
                    "phi_mean": [float(px), float(py)],
                    "flory_n_phases": n_phases,
                    "pass": n_phases == 3,
                }
            )

            if n_phases != 3:
                # Binodal three-phase energy via lever rule
                verts = np.array(poly.exterior.coords[:3])  # (3, 2) phase compositions
                phi_mean_2 = np.array([px, py])
                # Solve for volumes: V @ verts = phi_mean, sum(V) = 1
                A = np.vstack([verts.T, np.ones(3)])  # (3, 3)
                b = np.append(phi_mean_2, 1.0)
                vols_binodal = np.linalg.solve(A, b)
                f_binodal = float(
                    np.sum(
                        [
                            v * flory_energy(chis_reduced, verts[i])
                            for i, v in enumerate(vols_binodal)
                        ]
                    )
                )

                f_flory = float(
                    np.sum(
                        [
                            v * flory_energy(chis_reduced, phases.fractions[i, 1:])
                            for i, v in enumerate(phases.volumes)
                        ]
                    )
                )

                delta_f = f_binodal - f_flory
                # Flory finds lower energy => error in algorithm
                if delta_f > 0:
                    rec = records[-1]
                    rec["binodal_phis"] = verts.tolist()
                    rec["binodal_volumes"] = vols_binodal.tolist()
                    rec["flory_phis"] = phases.fractions[:, 1:].tolist()
                    rec["flory_volumes"] = phases.volumes.tolist()
                    rec["f_binodal"] = f_binodal
                    rec["f_flory"] = f_flory
                    rec["delta_f"] = delta_f

    return records


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate pyphasediagram against flory"
    )
    parser.add_argument(
        "-n",
        "--n-diagrams",
        type=int,
        default=100,
        help="Number of random chi matrices",
    )
    parser.add_argument(
        "--n-per-diagram",
        type=int,
        default=20,
        help="Total samples per diagram (split by phase-region area)",
    )
    parser.add_argument(
        "--tol", type=float, default=5e-2, help="Two-phase error tolerance"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per diagram in seconds (Unix only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="flory_validate_results.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    rng = np.random.default_rng(args.seed)
    results = []
    completed = 0

    pbar = tqdm(total=args.n_diagrams, desc="Diagrams")
    try:
        while completed < args.n_diagrams:
            chis_3x3, chis_reduced = random_chi_matrices(rng)

            _set_alarm(args.timeout)
            try:
                spinodal = Spinodal(chis_reduced)
                spinodal.build()
                binodal = Binodal(chis_reduced, spinodal.critical_points)
                binodal.build(unstable_manifold=spinodal.get_unstable_manifold())
                _cancel_alarm()
            except _Timeout:
                tqdm.write(
                    f"Timeout building diagram for chis_reduced =\n{chis_reduced}"
                )
                _cancel_alarm()
                continue
            except Exception as e:
                _cancel_alarm()
                tqdm.write(f"Error building diagram: {e}")
                continue

            if not binodal.binodal_sections:
                continue

            area_2p, area_3p = _compute_phase_areas(binodal)
            total_area = area_2p + area_3p
            if total_area > 0:
                n_2phase = round(args.n_per_diagram * area_2p / total_area)
                n_3phase = args.n_per_diagram - n_2phase
                # Guarantee at least 1 sample for each non-empty region
                if area_2p > 0 and n_2phase < 1:
                    n_2phase, n_3phase = 1, args.n_per_diagram - 1
                if area_3p > 0 and n_3phase < 1:
                    n_3phase, n_2phase = 1, args.n_per_diagram - 1
            else:
                n_2phase = args.n_per_diagram
                n_3phase = 0

            two_phase = validate_two_phase(
                binodal, chis_3x3, chis_reduced, rng, n_2phase, args.tol
            )
            three_phase = validate_three_phase(
                binodal, chis_3x3, chis_reduced, rng, n_3phase
            )

            n2_pass = sum(r["pass"] for r in two_phase)
            n2_total = len(two_phase)
            n3_pass = sum(r["pass"] for r in three_phase)
            n3_total = len(three_phase)

            entry = {
                "idx": completed,
                "chis_reduced": chis_reduced.tolist(),
                "chis_3x3": chis_3x3.tolist(),
                "n_sections": len(binodal.binodal_sections),
                "n_three_phase_polygons": len(binodal.three_phase_polygons),
                "two_phase": {
                    "passed": n2_pass,
                    "total": n2_total,
                    "samples": two_phase,
                },
                "three_phase": {
                    "passed": n3_pass,
                    "total": n3_total,
                    "samples": three_phase,
                },
            }
            results.append(entry)

            pbar.set_postfix(
                {"2ph": f"{n2_pass}/{n2_total}", "3ph": f"{n3_pass}/{n3_total}"}
            )
            pbar.update(1)
            completed += 1
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted \u2014 saving partial results...")
    finally:
        pbar.close()

    # Summary
    total_2p = sum(r["two_phase"]["total"] for r in results)
    pass_2p = sum(r["two_phase"]["passed"] for r in results)
    total_3p = sum(r["three_phase"]["total"] for r in results)
    pass_3p = sum(r["three_phase"]["passed"] for r in results)

    summary = {
        "n_diagrams": len(results),
        "two_phase_pass_rate": pass_2p / total_2p if total_2p else None,
        "two_phase_passed": pass_2p,
        "two_phase_total": total_2p,
        "three_phase_pass_rate": pass_3p / total_3p if total_3p else None,
        "three_phase_passed": pass_3p,
        "three_phase_total": total_3p,
    }

    output = {"summary": summary, "diagrams": results}
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path.resolve()}")
    print(f"Two-phase:   {pass_2p}/{total_2p} passed")
    print(f"Three-phase: {pass_3p}/{total_3p} passed")

    # Collect failures where flory found a lower energy
    flory_wins = []
    for r in results:
        for s in r["two_phase"]["samples"]:
            if not s["pass"] and s.get("delta_f") is not None and s["delta_f"] > 0:
                flory_wins.append(
                    {
                        "type": "two_phase",
                        "diagram_idx": r["idx"],
                        "chis_reduced": r["chis_reduced"],
                        "phi_mean": s["phi_mean"],
                        "binodal_phi_a": s["phi_a"],
                        "binodal_phi_b": s["phi_b"],
                        "flory_n_phases": s["flory_n_phases"],
                        "flory_phis": s["flory_phis"],
                        "flory_volumes": s["flory_volumes"],
                        "f_binodal": s["f_binodal"],
                        "f_flory": s["f_flory"],
                        "delta_f": s["delta_f"],
                    }
                )
        for s in r["three_phase"]["samples"]:
            if not s["pass"] and s.get("delta_f") is not None and s["delta_f"] > 0:
                flory_wins.append(
                    {
                        "type": "three_phase",
                        "diagram_idx": r["idx"],
                        "chis_reduced": r["chis_reduced"],
                        "phi_mean": s["phi_mean"],
                        "binodal_phis": s["binodal_phis"],
                        "binodal_volumes": s["binodal_volumes"],
                        "flory_n_phases": s["flory_n_phases"],
                        "flory_phis": s["flory_phis"],
                        "flory_volumes": s["flory_volumes"],
                        "f_binodal": s["f_binodal"],
                        "f_flory": s["f_flory"],
                        "delta_f": s["delta_f"],
                    }
                )

    failures_path = out_path.with_name(out_path.stem + "_flory_wins.json")
    failures_path.write_text(json.dumps(flory_wins, indent=2))
    print(f"Flory-wins:  {len(flory_wins)} cases written to {failures_path.resolve()}")

    if flory_wins:
        analyze_flory_wins(flory_wins)


# ---------------------------------------------------------------------------
# Analysis / visualisation of flory-wins
# ---------------------------------------------------------------------------
def _plot_phase_diagram(spinodal, binodal, ax):
    """Plot spinodal curve, binodal sections, and three-phase regions."""
    for i, comp in enumerate(nx.connected_components(spinodal.spinodal_graph)):
        sg = spinodal.spinodal_graph.subgraph(comp)
        phi1s, phi2s = spinodal._coords_from_subgraph(sg)
        ax.plot(phi1s, phi2s, "k--", label="Spinodal" if i == 0 else None)

    for i, section in enumerate(binodal.binodal_sections):
        ax.plot(
            section.phis[0, 0],
            section.phis[0, 1],
            "r-",
            label="Binodal" if i == 0 else None,
        )
        ax.plot(section.phis[1, 0], section.phis[1, 1], "b-")

    for j, poly in enumerate(binodal.three_phase_polygons):
        x, y = poly.exterior.xy
        ax.fill(
            x,
            y,
            color="red",
            alpha=0.3,
            label="3-phase region" if j == 0 else None,
        )

    ax.plot([0, 1], [1, 0], "k-", linewidth=0.5)
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel=r"$\phi_1$", ylabel=r"$\phi_2$", aspect="equal"
    )


def analyze_flory_wins(flory_wins: list[dict]):
    """Rebuild each flory-win diagram and plot binodal vs flory tie lines."""
    for i, fw in enumerate(flory_wins):
        chis_reduced = np.array(fw["chis_reduced"])

        spinodal = Spinodal(chis_reduced)
        spinodal.build()
        binodal = Binodal(chis_reduced, spinodal.critical_points)
        binodal.build(unstable_manifold=spinodal.get_unstable_manifold())

        _, ax = plt.subplots(figsize=(6, 6))
        _plot_phase_diagram(spinodal, binodal, ax)

        phi_mean = np.array(fw["phi_mean"])
        flory_phis = np.array(fw["flory_phis"])

        if fw["type"] == "two_phase":
            phi_a = np.array(fw["binodal_phi_a"])
            phi_b = np.array(fw["binodal_phi_b"])

            # Binodal tie line
            ax.plot(
                [phi_a[0], phi_b[0]],
                [phi_a[1], phi_b[1]],
                "g-o",
                markersize=6,
                linewidth=2,
                label="Binodal tie line",
            )
            # Flory tie line
            ax.plot(
                flory_phis[:, 0],
                flory_phis[:, 1],
                "mD-",
                markersize=6,
                linewidth=2,
                label="Flory tie line",
            )

        elif fw["type"] == "three_phase":
            binodal_phis = np.array(fw["binodal_phis"])
            # Binodal three-phase triangle
            tri_x = list(binodal_phis[:, 0]) + [binodal_phis[0, 0]]
            tri_y = list(binodal_phis[:, 1]) + [binodal_phis[0, 1]]
            ax.plot(
                tri_x, tri_y, "g-o", markersize=6, linewidth=2, label="Binodal 3-phase"
            )
            # Flory compositions
            ax.scatter(
                flory_phis[:, 0],
                flory_phis[:, 1],
                marker="D",
                s=60,
                color="magenta",
                edgecolors="k",
                zorder=11,
                label="Flory compositions",
            )

        # Mean composition
        ax.scatter(
            phi_mean[0],
            phi_mean[1],
            marker="x",
            s=120,
            color="orange",
            zorder=10,
            label=r"$\phi_{mean}$",
        )

        delta_f = fw["delta_f"]
        ax.set_title(
            f"Flory-win #{i+1} ({fw['type']})\n" rf"$\Delta f = {delta_f:.2e}$"
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
