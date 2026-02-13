import numpy as np
import networkx as nx


class Spinodal:

    def __init__(self, chis: np.ndarray):
        self.chis = chis
        self.spinodal_graph = nx.Graph()
        self.node_id = 0

    def build_spinodal(self):
        """
        Build the spinodal curve as a graph with nodes representing points (phi1, phi2) on the curve.
        Edges connect consecutive points along the curve. The graph is stored in self.spinodal_graph.
        """
        phi1_domains = self._spinodal_domains()
        for i in range(0, len(phi1_domains), 2):
            self._domain_data(phi1_domains[i], phi1_domains[i + 1])
        self._clip_to_domain()
        self._connect_branches()

    def _get_p_q(self, phi, is_calculate_phi2=True):
        """
        Compute the coefficients p and q of the quadratic equation for phi1/phi2 given phi2/phi1.
        """
        a, b, c = self.chis[0, 0], self.chis[1, 1], self.chis[0, 1]
        if is_calculate_phi2 == False:
            a, b = b, a  # swap a and b if computing phi1 from phi2
        det = a * b - c**2
        p = (2 * phi * c - a - phi * (1 - phi) * det) / (a + det * phi)
        q = (1 + phi * (1 - phi) * b) / (a + det * phi)
        return p, q

    def phi2_from_phi1(self, phi1):
        """Given phi1, compute the two possible phi2 values from the quadratic formula."""
        p, q = self._get_p_q(phi1)
        discriminant = p**2 / 4 + q

        return -p / 2 + np.sqrt(discriminant), -p / 2 - np.sqrt(discriminant)

    def _domain_data(self, phi1_i, phi1_f):
        """
        Compute the spinodal branches for phi1 in [phi1_i, phi1_f]
        and add them to the graph as nodes with attributes phi1, phi2, and pos=(phi1, phi2).
        """
        num_points = 1000
        dphi1 = (phi1_f - phi1_i) / num_points
        phi1_vals = np.linspace(phi1_i, phi1_f, num_points)
        start_id = self.node_id
        for i, phi1 in enumerate(phi1_vals):
            phi2_pos, phi2_neg = self.phi2_from_phi1(phi1)
            for phi2 in [phi2_pos, phi2_neg]:
                self.spinodal_graph.add_node(
                    self.node_id, phi1=phi1, phi2=phi2, pos=(phi1, phi2)
                )
                if i:
                    self.spinodal_graph.add_edge(self.node_id - 2, self.node_id)
                self.node_id += 1

            # Connect left end of branches if they are not at a pole
            # if i == 0 and phi1_i > 0 and abs(phi2_pos - phi2_neg) < 1e-1:

            #     self.spinodal_graph.add_edge(self.node_id - 2, self.node_id - 1)
            # Connect right end of branches if they are not at a pole
            # if i == len(phi1_vals) - 1 and abs(phi2_pos - phi2_neg) < 1e-1:
            #     self.spinodal_graph.add_edge(self.node_id - 2, self.node_id - 1)
        # for i in [start_id, self.node_id-2]:
        #     posi = self.spinodal_graph.nodes[i]["pos"]
        #     for j in [start_id+1, self.node_id-1]:
        #         posj = self.spinodal_graph.nodes[j]["pos"]
        #         if posi

    def _spinodal_domains(self):
        """
        Find phi values where the discriminant of the quadratic equation for phi2 changes sign.
        These are the phi1 coordinates where the spinodal branches start/end
        """
        phi1_vals = np.linspace(0, 1, 100000)

        p, q = self._get_p_q(phi1_vals)
        discriminant = p**2 + 4 * q

        roots_idx = np.where(np.diff(np.sign(discriminant)))[0]

        # Handle edge cases where the discriminant is positive at the endpoints
        if discriminant[0] > 0:
            roots_idx = np.r_[0, roots_idx]
        if discriminant[-1] > 0:
            roots_idx = np.r_[roots_idx, -1]
        # Subtract 1 from every other root index to get the start of the domain where discriminant is positive
        roots_idx[::2] += 1
        roots_idx[1::2] -= 1
        return phi1_vals[roots_idx]

    def _clip_to_domain(self):
        """
        Remove nodes whose (phi1, phi2) lie outside the correct domain
        Mutates and returns G for convenience.
        """
        to_remove = [
            n
            for n, d in self.spinodal_graph.nodes(data=True)
            if (d["phi1"] < 0) or (d["phi2"] < 0) or (d["phi1"] + d["phi2"] > 1)
        ]
        self.spinodal_graph.remove_nodes_from(to_remove)

    def _connect_branches(self):
        if self.spinodal_graph.number_of_nodes() == 0:
            return
        comps = list(nx.connected_components(self.spinodal_graph))
        for i, compi in enumerate(comps):
            graphi = self.spinodal_graph.subgraph(compi)
            nodei_list = [
                n for n in graphi.nodes(data=True) if graphi.degree[n[0]] == 1
            ]
            closest_node_list = [0] * len(nodei_list)
            closest_dist_list = [float("inf")] * len(nodei_list)

            for j, compj in enumerate(comps):
                graphj = self.spinodal_graph.subgraph(comps[j])
                nodej_list = [
                    n for n in graphj.nodes(data=True) if graphj.degree[n[0]] == 1
                ]
                for ni, nodei in enumerate(nodei_list):
                    posi = nodei[1]["pos"]
                    for nj, nodej in enumerate(nodej_list):
                        posj = nodej[1]["pos"]
                        dist = np.sqrt(
                            (posi[0] - posj[0]) ** 2 + (posi[1] - posj[1]) ** 2
                        )
                        if dist < closest_dist_list[ni] and nodei[0] != nodej[0]:
                            closest_dist_list[ni] = dist
                            closest_node_list[ni] = nodej[0]
            for ni, nodei in enumerate(nodei_list):
                if closest_dist_list[ni] < 0.05:
                    self.spinodal_graph.add_edge(nodei[0], closest_node_list[ni])
                    return self._connect_branches()


def apply_per_component(
    G: nx.Graph,
    func,
):
    """
    Apply func(x,y) -> (u,v) to every node.
    Return: list_of_lists, where each inner list corresponds to one connected component
            and contains the function outputs for nodes in that component.

    Notes:
    - Components are returned in arbitrary order.
    - Node order inside each component is arbitrary; if you want a geometric order
      along the curve, see the optional ordering snippet below.
    """
    component_outputs = []
    for comp_nodes in nx.connected_components(G):
        outputs = []
        for n in comp_nodes:
            d = G.nodes[n]
            outputs.append(func(d["x"], d["y"]))
        component_outputs.append(outputs)
    return component_outputs


# ---------------------------
# Example usage
# ---------------------------
chi_dr, chi_rs, chi_ds = [np.random.random() + 2] * 3 + np.random.random(3) / 100
# chi_dr, chi_rs, chi_ds = 2.6665467662664977, 2.6722868906613435, 2.672457833788111

chi_11, chi_22, chi_12 = -2 * chi_ds, -2 * chi_rs, chi_dr - chi_rs - chi_ds
chis = np.array([[chi_11, chi_12], [chi_12, chi_22]])
spinodal = Spinodal(chis)
spinodal.build_spinodal()

print(
    f"Spinodal graph has {spinodal.spinodal_graph.number_of_nodes()} nodes and number of subgraphs: {nx.number_connected_components(spinodal.spinodal_graph)}"
)
import matplotlib.pyplot as plt
import networkx as nx


def plot_components_native(G):
    plt.figure()

    num_comps = nx.number_connected_components(G)
    for i, comp in enumerate(nx.connected_components(G)):
        H = G.subgraph(comp)

        # Find endpoint (degree 1) if it exists (path case)
        endpoints = [n for n, d in H.degree() if d == 1]

        if endpoints:
            start = endpoints[0]
        else:
            # cycle case (all degree 2)
            start = next(iter(H.nodes()))

        ordered_nodes = list(nx.dfs_preorder_nodes(H, source=start))

        xs = [H.nodes[n]["phi1"] for n in ordered_nodes]
        ys = [H.nodes[n]["phi2"] for n in ordered_nodes]
        if np.isnan(ys).any():
            print("NaN values found in ys, skipping plot for this component.")
        print(f"Length of component {i}: {len(ordered_nodes)}")
        plt.plot(xs, ys)
        plt.scatter(xs[0], ys[0], alpha=0.45)
        plt.scatter(xs[-1], ys[-1], alpha=0.45)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [1, 0], "k--")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Disjoint components (ordered via NetworkX DFS)")
    plt.show()


plot_components_native(spinodal.spinodal_graph)
