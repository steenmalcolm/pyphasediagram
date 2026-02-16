# binodal.py
import networkx as nx
import numpy as np
from pyphasediagram.stepper.point import CriticalPoint, BinodalPoint


class Binodal:

    def __init__(self, chis: np.ndarray, critical_points: list[BinodalPoint]):
        """Initialize the Binodal class with a 2x2 matrix of chi parameters"""
        self.chis = chis
        self.binodal_graph = nx.Graph()
        self.node_id = 0
        self.critical_points = []

    def build(self):
        """
        Build the binodal curve as a graph with nodes representing points (phi1a, phi2a, phi1b, phi2b) on the curve.
        Edges connect consecutive points along the curve. The graph is stored in self.binodal_graph.
        """
        pass
