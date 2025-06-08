from utils.graph import DecentralizedNetwork

dec_graph = DecentralizedNetwork(num_nodes=4, graph_type='ring', dim=1, seed=42)

print("Directed Graph nodes:", dec_graph.G_directed.nodes())
print("Directed Graph edges:", dec_graph.G_directed.edges())
print("Arc mapping:", dec_graph.arc_mapping)
print("Neighbors:", dec_graph.neighbors)
print("Degrees:", dec_graph.degrees)

print("A1:", dec_graph.A1)
print("A2:", dec_graph.A2)
print("M_plus:", dec_graph.M_plus)
print("M_minus:", dec_graph.M_minus)
print("L_plus:", dec_graph.L_plus)
print("L_minus:", dec_graph.L_minus)
print("W:", dec_graph.W)

dec_graph.visualize_graph()