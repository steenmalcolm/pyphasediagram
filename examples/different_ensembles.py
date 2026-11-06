"""Different ensembles of a ternary phase diagram."""

import matplotlib.pyplot as plt
import numpy as np

from pyphasediagram import PhaseDiagram, utils


# interaction matrix (chi parameters) for a 
# ternary mixture of solvent, regulator, and droplet material,
# reproducing a phase diagram from the preprint:
# Köstler, Steen, Zwicker, https://arxiv.org/abs/2609.24529 (2026)
chis = np.array([[0, 1.7, 2.5],     # phi_S solvent
                 [1.7, 0, 1.3],     # phi_R regulator
                 [2.5, 1.3, 0]])    # phi_D droplet material

# compute the phase diagram
diagram = PhaseDiagram(chis)
diagram.build()


fig = plt.figure(figsize=(20, 4))
fig.subplots_adjust(wspace=.5, left=0.05, right=0.95, bottom=0.15, top=0.85)


# phase diagram in the canonical ensemble (ternary plot)
ax0 = plt.subplot(1, 4, 1,projection="ternary")
# phase diagram in the canonical ensemble
ax1 = plt.subplot(1, 4, 2)
# phase diagram in the semi-grand-canonical ensemble
ax2 = plt.subplot(1, 4, 3)
# phase diagram in the grand-canonical ensemble
ax3 = plt.subplot(1, 4, 4)


# spinodal
import networkx as nx

for i, comp in enumerate(nx.connected_components(diagram.spinodal.spinodal_graph)):
    sg = diagram.spinodal.spinodal_graph.subgraph(comp)
    phi1s, phi2s = diagram.spinodal._coords_from_subgraph(sg)
    label = "Spinodal" if i == 0 else None
    ax0.plot(
        *diagram._to_ternary_coordinates(phi1s, phi2s),
        "k--",
        label=label,
    )
    ax1.plot(
        phi1s, phi2s,
        "k--",
        label=label,
    )
    ax2.plot(
        utils.exchange_chemical_potentials((phi1s, phi2s), chis)[0], phi2s,
        "k--",
        label=label,
    )
    ax3.plot(
        *utils.exchange_chemical_potentials((phi1s, phi2s), chis),
        "k--",
        label=label,
    )

# binodal sections
for i, section in enumerate(diagram.binodal.binodal_sections):
    label = "Binodal" if i == 0 else None
    ax0.plot(
        *diagram._to_ternary_coordinates(section.phis[0, 0], section.phis[0, 1]),
        color="green",
        linewidth=2.5,
        label=label,
    )
    ax0.plot(
        *diagram._to_ternary_coordinates(section.phis[1, 0], section.phis[1, 1]),
        linewidth=2.5,
        color="green",
    )
    ax1.plot(
        section.phis[0, 0], section.phis[0, 1],
        color="green",
        linewidth=2.5,
        label=label,
    )
    ax1.plot(
        section.phis[1, 0], section.phis[1, 1],
        linewidth=2.5,
        color="green",
    )
    ax2.plot(
        utils.exchange_chemical_potentials((section.phis[0, 0], section.phis[0, 1]), chis)[0], section.phis[0, 1],
        color="green",
        linewidth=2.5,
        label=label,
    )
    ax2.plot(
        utils.exchange_chemical_potentials((section.phis[1, 0], section.phis[1, 1]), chis)[0], section.phis[1, 1],
        linewidth=2.5,
        color="green",
    )
    # in the grand-canonical ensemble, the binodal is a single curve 
    # in the chemical potential space, so we only need to plot one of the two branches
    ax3.plot(
        *utils.exchange_chemical_potentials((section.phis[0, 0], section.phis[0, 1]), chis),
        color="green",
        linewidth=2.5,
        label=label,
    )


# critical points
for i, cp in enumerate(diagram.spinodal.critical_points):
    label = "Critical point" if i == 0 else None
    ax0.scatter(
        *diagram._to_ternary_coordinates(cp.phi1, cp.phi2),
        marker="*",
        color="gold",
        edgecolors="k",
        s=100,
        zorder=6,
        label=label,
    )
    ax1.scatter(
        cp.phi1, cp.phi2,
        marker="*",
        color="gold",
        edgecolors="k",
        s=100,
        zorder=6,
        label=label,
    )
    ax2.scatter(
        utils.exchange_chemical_potentials((cp.phi1, cp.phi2), chis)[0], cp.phi2,
        marker="*",
        color="gold",
        edgecolors="k",
        s=100,
        zorder=6,
        label=label,
    )
    ax3.scatter(
        *utils.exchange_chemical_potentials((cp.phi1, cp.phi2), chis),
        marker="*",
        color="gold",
        edgecolors="k",
        s=100,
        zorder=6,
        label=label,
    )

ax0.set_title("Canonical ensemble (ternary plot)")
ax0.set_tlabel(r"$\phi_S$", fontsize=16)
ax0.set_llabel(r"$\phi_R$", fontsize=16)
ax0.set_rlabel(r"$\phi_D$", fontsize=16)

ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_title("Canonical ensemble")
ax1.set_xlabel(r"$\phi_R$", fontsize=16)
ax1.set_ylabel(r"$\phi_D$", fontsize=16)
ax1.legend(loc="upper right", fontsize=12)
ax1.set_box_aspect(1)

ax2.set_title("Semi-grand-canonical ensemble")
ax2.set_xlim(-5,0)
ax2.set_ylim(0,1)
ax2.set_xlabel(r"$\bar\mu_R$", fontsize=16)
ax2.set_ylabel(r"$\phi_D$", fontsize=16)
ax2.set_box_aspect(1)

ax3.set_title("Grand-canonical ensemble")
ax3.set_xlim(-5,0)
ax3.set_xlabel(r"$\bar\mu_R$", fontsize=16)
ax3.set_ylabel(r"$\bar\mu_D$", fontsize=16)
ax3.set_box_aspect(1)



plt.show()


