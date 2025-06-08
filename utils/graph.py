import numpy as np
import networkx as nx

class DecentralizedNetwork:
    def __init__(self, num_nodes: int=2, dim: int = 1, graph_type: str='ring', seed: int=42, 
                    p: float=0.3, degree: int=2):
        """
        Initialize a Graph object.
        
        Args:
            num_nodes (int): Number of nodes in the graph. (L)
            dim (int): Dimension of the data associated with each node. (N)
            graph_type (str): Type of the graph (ring, complete, line, random, degree).
            seed (int): Random seed for reproducibility.
            p (float): Probability of edge creation for random graphs.
        """
        self.num_nodes = num_nodes
        self.graph_type = graph_type
        self.seed = seed
        self.dim = dim
        self.p = p
        self.degree = degree

        if num_nodes <= 0:
            raise ValueError("Number of nodes must be a positive integer.")
        
        self.generate_graph()
        self.generate_admm_matrix()
        

    def generate_graph(self):
        """
        Generate a random graph based on the specified parameters.

        G_directed: symmetric directed graph with edges in both directions.
        edges: number of edges in the graph. (E)
        arc_mapping: mapping from directed edge (i, j) to arc index q. (q_i)
        neighbors: dictionary mapping each node to its neighbors.
        degrees: dictionary mapping each node to its degree.
        """
        
        np.random.seed(self.seed)
        # nx.seed(self.seed)

        if self.graph_type == 'ring':
            G_undir = nx.cycle_graph(self.num_nodes)
        
        elif self.graph_type == 'complete':
            G_undir = nx.complete_graph(self.num_nodes)
        
        elif self.graph_type == 'line':
            G_undir = nx.path_graph(self.num_nodes)
        
        elif self.graph_type == 'random':
            G_undir = nx.erdos_renyi_graph(self.num_nodes, self.p, seed=self.seed)
        
        elif self.graph_type == 'degree':
            if self.degree >= self.num_nodes:
                raise ValueError("Degree must be less than the number of nodes for degree-based graph generation.")
            elif self.degree is None or self.degree < 2:
                raise ValueError("Degree must be a positive integer greater than 1.")
            
            if self.degree == 2:
                G_undir = nx.cycle_graph(self.num_nodes)
            else:
                G_undir = nx.random_regular_graph(self.degree, self.num_nodes, seed=self.seed)
        
        else:
            raise ValueError("Unsupported graph type.")
        
        # check if the graph is connected
        if not nx.is_connected(G_undir):
            raise ValueError("Generated graph is not connected.")
        
        # convert to symmetric directed graph (default graphs are undirected)
        self.G_directed = nx.DiGraph()
        self.G_directed.add_nodes_from(G_undir.nodes())
        self.edges = len(G_undir.edges())
        arc_index = 0
        self.arc_mapping = {}
        for i, j in G_undir.edges():
            self.G_directed.add_edge(i, j)
            self.G_directed.add_edge(j, i)
            self.arc_mapping[(i, j)] = arc_index
            arc_index += 1
            self.arc_mapping[(j, i)] = arc_index #+ len(G_undir.edges())
            arc_index += 1

        # compute neighbors dict and degree for each node
        self.neighbors = {node: list(G_undir.neighbors(node)) for node in G_undir.nodes()}
        self.degrees = {node: G_undir.degree(node) for node in G_undir.nodes()}

    
    def generate_admm_matrix(self):
        """
        Generate the dec ADMM related matrix for the decentralized network.
        
        A1: source incidence matrix of the directed graph. (2EN x LN)
        A2: target incidence matrix of the directed graph. (2EN x LN)

        M_plus: extended unoriented incidence matrix. A1.T + A2.T 
        M_minus: extended oriented incidence matrix. A1.T - A2.T 

        L_plus: extended signless laplacian matrix. 1/2 * (M+ @ M-.T) (LN x LN)
        L_minus: extended signed laplacian matrix. 1/2 * (M- @ M-.T) (LN x LN)

        W: weight matrix for the decentralized network. diag(|N_i|*I_N)  (LN x LN)
        where |N_i| is the number of neighbors of node i and 
        I_N is the identity matrix of dimension N.
        """        
        # incidence matrix A1, A2
        self.A1 = np.zeros((2 * self.edges * self.dim, self.num_nodes * self.dim))
        self.A2 = np.zeros((2 * self.edges * self.dim, self.num_nodes * self.dim))

        for (i, j), q in self.arc_mapping.items():
            row_start = q * self.dim
            
            col_start = i * self.dim
            self.A1[row_start:row_start + self.dim, col_start:col_start + self.dim] = np.eye(self.dim)

            col_start = j * self.dim
            self.A2[row_start:row_start + self.dim, col_start:col_start + self.dim] = np.eye(self.dim)

        # extented unoriented / oriented incidence matrix M+, M-
        self.M_plus = self.A1.T + self.A2.T
        self.M_minus = self.A1.T - self.A2.T

        # extended signless / signed laplacian matrix L+, L-
        self.L_plus = 0.5 * (self.M_plus @ self.M_plus.T)
        self.L_minus = 0.5 * (self.M_minus @ self.M_minus.T)

        # W matrix
        self.W = np.zeros((self.num_nodes * self.dim, self.num_nodes * self.dim))
        for i in range(self.num_nodes):
            self.W[i * self.dim:(i + 1) * self.dim, i * self.dim:(i + 1) * self.dim] = self.degrees[i] * np.eye(self.dim)
                

    def visualize_graph(self):
        """
        Visualize the graph using matplotlib.
        """
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(self.G_directed)
        nx.draw(self.G_directed, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.title(f"{self.graph_type.capitalize()} Graph with {self.num_nodes} Nodes")
        plt.show()
        
        





    


    