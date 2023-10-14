from rdflib import Graph, URIRef, Literal, Namespace
from collections import deque

from sa_helper import *

def spreading_activation_BFS(graph, G, seed_entities, neighbor_counts, alpha, beta, max_hops,
                             extraction_threshold, strict=True, fan_out=True, excl=True, pop=True):
    """
    Main function that performs the Spreading Activation process based on 
    BFS (Breadth-First Search) on an RDF graph.

    Args:
        graph (rdflib.Graph): The RDF graph.
        G (nx.DiGraph): The NetworkX directed graph.
        seed_entities (list of str): The URIs of the seed entities to start the SA process.
        neighbor_counts (dict): A dictionary that contains the neighbor node number for each
            nodes in the graph.
        alpha (float): The distance decay parameter.
        beta (float): The firing threshold.
        max_hops (int): The maximum number of hops or connections that can be followed from 
            the starting node to reach another node.
        extraction_threshold (float): The threshold for including entities in the subgraph.
        strict (bool): If True, the extraction of entities will be strict. Defaults to True.
        fan_out (bool): If True, fan-out constraint will be utilized in the output function
            so that it equally divides the available activation value by the outgoing edges.
            Defaults to True.
        excl (bool): If True, exclusivity factor is used to calculate the accessibility of 
            the node for the given relation. Defaults to True.
        pop (bool): If True, popularity factor is used to calculate the accessibility of 
            the node for the given relation. Defaults to True.
        
    Returns:
        Graph: The resulting subgraph after the Spreading Activation process.
    """
    ACTIVATION = Namespace("http://example.org/activation/")
    activation_predicate = ACTIVATION['value']  
    
    # Collect all entities in the graph
    entities = set()
    for s, _, o in graph:
        entities.add(s)
        entities.add(o)
    
    # Initialize the activation value of all entities to 0
    for entity in entities:
        graph.set((entity, activation_predicate, Literal(0)))

    # Turn list of str into list of uri
    seed_entities_uri = [URIRef(seed) for seed in seed_entities]
    
    # Initialize hops
    for s, o in G.edges():
        G[s][o]['hops'] = 0

    # Set the activation value of seed entities to 1 and initialize hop counts
    for seed_uri in seed_entities_uri:
        graph.set((seed_uri, activation_predicate, Literal(1)))
        for neighbor in G.neighbors(seed_uri):
            G[seed_uri][neighbor]['hops'] = 1

    # Calculate the total number of nodes in the graph
    num_all_nodes = len(G.nodes())
    
    # Use a queue for BFS traversal
    queue = deque(seed_entities_uri)
    visited = set(seed_entities_uri)  # Record visited nodes

    while queue:
        current_node = queue.popleft()
        current_activation_value = float(graph.value(subject=current_node, predicate=activation_predicate)) if graph.value(subject=current_node, predicate=activation_predicate) is not None else 0

        # If the current node's activation value is above the threshold, operate on its neighbors
        if current_activation_value > beta:
            neighbors = list(G.neighbors(current_node))
            
            for neighbor in neighbors:
                # hop count restriction
                if G[current_node][neighbor]['hops'] > max_hops:
                    continue
                
                # Increment hop count for the all neighbor nodes of current node's neighbor node
                neigh_neighbors = list(G.neighbors(neighbor))
                for nn in neigh_neighbors:
                    G[neighbor][nn]['hops'] = G[current_node][neighbor]['hops'] + 1
                
                #Gain the neighbor node number of each node
                node_neighbor_count = neighbor_counts[current_node]
                
                current_neighbor_activation = float(graph.value(subject=neighbor, predicate=activation_predicate)) if graph.value(subject=neighbor, predicate=activation_predicate) is not None else 0
     
                # Calculate the input activation value for each current node based on the output activation value of its neighbor nodes  
                entity_input = input_function(graph, G, neighbor, alpha, beta, activation_predicate, node_neighbor_count, fan_out, excl, pop)

                # Calculate the new activation for the entity
                new_activation_value = activation_function(current_neighbor_activation, entity_input)

                # Update activation value
                graph.set((neighbor, activation_predicate, Literal(new_activation_value)))

                # If the neighbor has not been visited, add it to the queue
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)                

    # Initialize a list to collect subgraph triples
    subgraph_triples = []

    # Iterate through the graph to find entities with activation value above the threshold
    for s, p, o in graph:
        s_activation_value = float(graph.value(subject=s, predicate=activation_predicate)) if graph.value(subject=s, predicate=activation_predicate) is not None else 0
        o_activation_value = float(graph.value(subject=o, predicate=activation_predicate)) if graph.value(subject=o, predicate=activation_predicate) is not None else 0
        
        if strict:
            if s_activation_value > extraction_threshold and o_activation_value > extraction_threshold:
                subgraph_triples.append((s, p, o))
        else:
            if s_activation_value > extraction_threshold or o_activation_value > extraction_threshold:
                subgraph_triples.append((s, p, o))
    
    import csv
    with open('activation_values_per_triple.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Subject", "Predicate", "Object", "Subject Activation Value", "Object Activation Value"])
        for s, p, o in subgraph_triples:
            s_activation_value = float(graph.value(subject=s, predicate=activation_predicate)) if graph.value(subject=s, predicate=activation_predicate) is not None else 0
            o_activation_value = float(graph.value(subject=o, predicate=activation_predicate)) if graph.value(subject=o, predicate=activation_predicate) is not None else 0
            csvwriter.writerow([s, p, o, s_activation_value, o_activation_value])
            
    # Create the resulting subgraph from the collected triples
    subgraph = Graph()
    for triple in subgraph_triples:
        subgraph.add(triple)

    return subgraph


def spreading_activation_DFS(graph, G, seed_entities, neighbor_counts, alpha, beta, max_depth,
                             extraction_threshold, strict=True, fan_out=True, excl=True, pop=True):
    """
    Main function that performs the Spreading Activation process based on 
    DFS (Depth-First Search) on an RDF graph.

    Args:
        graph (rdflib.Graph): The RDF graph.
        G (nx.DiGraph): The NetworkX directed graph.
        seed_entities (list of str): The URIs of the seed entities to start the Spreading Activation process.
        neighbor_counts (dict): A dictionary that contains the neighbor node number for each
            nodes in the graph.
        alpha (float): The distance decay parameter.
        beta (float): The firing threshold.
        max_depth (int): The maximum depth of the DFS traversal for the SA process.
        extraction_threshold (float): The threshold for including entities in the subgraph.
        strict (bool): If True, the extraction of entities will be strict. Defaults to True.
        fan_out (bool): If True, fan-out constraint will be utilized in the output function
            so that it equally divides the available activation value by the outgoing edges.
            Defaults to True.
        excl (bool): If True, exclusivity factor is used to calculate the accessibility of the node 
            for the given relation. Defaults to True.
        pop (bool): If True, popularity factor is used to calculate the accessibility of the node 
            for the given relation. Defaults to True.
        
    Returns:
        Graph: The resulting subgraph after the Spreading Activation process.
    """
    ACTIVATION = Namespace("http://example.org/activation/")
    activation_predicate = ACTIVATION['value']

    # Collect all entities in the graph
    entities = set()
    for s, _, o in graph:
        entities.add(s)
        entities.add(o)
    
    # Initialize the activation value of all entities to 0
    for entity in entities:
        graph.set((entity, activation_predicate, Literal(0)))
    
    # Calculate the total number of nodes in the graph
    num_all_nodes = len(G.nodes())
    
    # Turn list of str into list of uri
    seed_entities_uri = [URIRef(seed) for seed in seed_entities]
    
    # Set the activation value of seed entities to 1 
    for seed_uri in seed_entities_uri:
        graph.set((seed_uri, activation_predicate, Literal(1)))

        # Perform Depth-First Search from the seed entity
        stack = [(seed_uri, 0)]  # Stack to hold nodes and their depth count
        visited = set()  # Track visited nodes

        while stack:
            current_node, depth = stack.pop()
            if current_node in visited or depth > max_depth:
                continue

            visited.add(current_node)
            current_activation_value = float(graph.value(subject=current_node, predicate=activation_predicate)) if graph.value(subject=current_node, predicate=activation_predicate) is not None else 0

            #Gain the neighbor node number of each node
            node_neighbor_count = neighbor_counts[current_node]
            # Calculate the input activation value for each current node based on the output activation value of its neighbor nodes 
            entity_input = input_function(graph, G, current_node, alpha, beta, activation_predicate, node_neighbor_count, fan_out, excl, pop)

            # Calculate the new activation for the entity
            new_activation_value = activation_function(current_activation_value, entity_input)

            # Update the activation value in the graph
            graph.set((current_node, activation_predicate, Literal(new_activation_value)))

            # Add neighbors to the stack with incremented depth count
            for neighbor in G.successors(current_node):
                stack.append((neighbor, depth + 1))

    # Initialize a list to collect subgraph triples
    subgraph_triples = []

    # Iterate through the graph to find entities with activation value above the threshold
    for s, p, o in graph:
        s_activation_value = float(graph.value(subject=s, predicate=activation_predicate)) if graph.value(subject=s, predicate=activation_predicate) is not None else 0
        o_activation_value = float(graph.value(subject=o, predicate=activation_predicate)) if graph.value(subject=o, predicate=activation_predicate) is not None else 0
        
        if strict:
            if s_activation_value > extraction_threshold and o_activation_value > extraction_threshold:
                subgraph_triples.append((s, p, o))
        else:
            if s_activation_value > extraction_threshold or o_activation_value > extraction_threshold:
                subgraph_triples.append((s, p, o))
    
    import csv
    with open('activation_values_per_triple.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Subject", "Predicate", "Object", "Subject Activation Value", "Object Activation Value"])
        for s, p, o in subgraph_triples:
            s_activation_value = float(graph.value(subject=s, predicate=activation_predicate)) if graph.value(subject=s, predicate=activation_predicate) is not None else 0
            o_activation_value = float(graph.value(subject=o, predicate=activation_predicate)) if graph.value(subject=o, predicate=activation_predicate) is not None else 0
            csvwriter.writerow([s, p, o, s_activation_value, o_activation_value])
            
    # Create the resulting subgraph from the collected triples
    subgraph = Graph()
    for triple in subgraph_triples:
        subgraph.add(triple)

    return subgraph

