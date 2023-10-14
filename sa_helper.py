import math
from rdflib import Graph, URIRef
import networkx as nx

def output_function(activation_value, alpha, beta, neighbor_count, fan_out=False):
    """
    Function that calculates the output activation for an entity.

    Args:
        activation_value (float): The current activation value of the entity.
        alpha (float): The distance decay parameter that decays the activation going out 
            of each node exponentially with respect to pulse.
        beta (float): The firing threshold that limits the nodes that fire to those nodes 
            whose activation value surpasses this given threshold for the first time.
        neighbor_count (int): The neighbor node number of the current node.
        fan_out (bool): If True, fan-out constraint will be utilized in the output function
            so that it equally divides the available activation value by the outgoing edges.
            Defaults to False.
            
    Returns:
        float: The output activation value of the entity.
    """
    if activation_value > beta:
        out_activation_value = activation_value * alpha
        if fan_out:
            out_activation_value = out_activation_value / neighbor_count if neighbor_count > 0 else 0
        
        return out_activation_value
    
    else:
        
        return 0


def calculate_excl(G, node, neighbor_node, relation):
    """
    This function calculates the exclusivity of a relation in the graph.
    It is calculated as the inverse of the frequency of the relation in the graph.
    
    Args:
        G (nx.DiGraph): The NetworkX directed graph.
        node (rdflib.term.URIRef): The node for which to calculate the exclusivity.
        neighbor_node (rdflib.term.URIRef): The neighbor node for which to calculate the exclusivity.
        relation (rdflib.term.URIRef): The relation for which to calculate the exclusivity.

    Returns:
        float: The exclusivity of the relation.
    """
    # Count the number of relation that exit node 
    count_r_out = 0
    
    for _, edge_data in G[node].items():
        if edge_data['predicate'] == relation:
            count_r_out +=1
            
    # Count the number of relation that enter node (a neighbor node)
    count_r_in = 0
    
    for _, _, edge_data in G.in_edges(neighbor_node, data=True):
        if edge_data['predicate'] == relation:
            count_r_in += 1

    # Calculate the exclusivity of the relation
    excl = 1 / (count_r_out + count_r_in - 1)

    return excl


def calculate_pop(graph, G, node, num_all_nodes):
    """
    This function calculates the popularity of a node in the graph.
    It is calculated as log(D(c)) / log(|V| - 1), where D(c) is the number of neighbors of the node,
    and |V| is the total number of nodes in the graph.
    
    Args:
        graph (rdflib.Graph): The RDF graph.
        G (nx.DiGraph): The NetworkX directed graph.
        node (rdflib.term.URIRef): The node for which to calculate the popularity. 
        num_all_nodes(int): The total number of nodes in the graph.
        
    Returns:
        float: The popularity of the node.
    """ 
    # Calculate D(c)
    num_neighbors = len(list(G.neighbors(node)))
    
    # Check for num_neighbors = 0 to avoid log(0)
    if num_neighbors == 0:
        return 0

    # Calculate the popularity of the node
    pop = math.log(num_neighbors) / math.log(num_all_nodes - 1) if num_all_nodes > 1 else 0

    return pop


def input_function(graph, G, node, alpha, beta, activation_predicate, neighbor_count, num_all_nodes,
                   fan_out=False, excl=True, pop=True):
    """
    Function that calculates the input activation for an entity.
    
    Args:
        graph (rdflib.Graph): The RDF graph.
        G (nx.DiGraph): The NetworkX directed graph.
        node (rdflib.term.URIRef): The node for which to calculate the input activation value.
        alpha (float): The distance decay parameter that decays the activation going out 
            of each node exponentially with respect to pulse.
        beta (float): The firing threshold that limits the nodes that fire to those nodes 
            whose activation value surpasses this given threshold for the first time.
        activation_predicate (rdflib.term.URIRef): The predicate used to store activation values.
        neighbor_count (int): The neighbor node number of the current node.
        num_all_nodes(int): The total number of nodes in the graph.
        fan_out (bool): If True, fan-out constraint will be utilized in the output function
            so that it equally divides the available activation value by the outgoing edges.
            Defaults to False.
        excl (bool): If True, exclusivity factor is used to calculate the accessibility of the node 
            for the given relation. Defaults to True.
        pop (bool): If True, popularity factor is used to calculate the accessibility of the node 
            for the given relation. Defaults to True.

    Returns:
        float: The input activation value of the entity.
    """
    # Initialize the total accr value for all neighbors and the input activation value
    total_accr_neighbors = 0
    in_activation_value = 0

    # Iterate through all the neighbors of the node
    for neighbor, edge_data in G[node].items():
        # Get the relation (predicate) between the node and the neighbor
        relation = edge_data['predicate']
        #four accessibility computation schemes according to the exclusivity and popularity factors (ExclPop) being used or not
        if excl == True and pop == True:
            # Calculate the accessibility of the neighbor for the given relation
            accr_neighbor = calculate_excl(G, node, neighbor, relation) * calculate_pop(graph, G, neighbor, num_all_nodes)
            # Calculate the accessibility of the node for the given relation
            accr_node = calculate_excl(G, node, neighbor, relation) * calculate_pop(graph, G, node, num_all_nodes)
        
        elif excl == True and pop == False:
            accr_neighbor = accr_node = calculate_excl(G, node, neighbor, relation)
            
        elif excl == False and pop == True:  
            accr_neighbor = calculate_pop(graph, G, neighbor, num_all_nodes)
            accr_node = calculate_pop(graph, G, node, num_all_nodes)
            
        else:
            accr_neighbor = accr_node = 1.0
            
        # Get the neighbor's activation value
        neighbor_activation_value = graph.value(subject=neighbor, predicate=activation_predicate)
        neighbor_activation_value = float(neighbor_activation_value) if neighbor_activation_value is not None else 0

        # Calculate the input activation value
        in_activation_value += output_function(neighbor_activation_value, alpha, beta, neighbor_count, fan_out) * accr_node
        
        # Add the accessibility value to the total accr value for neighbors
        total_accr_neighbors += accr_neighbor

    # Normalize by the total accessibility of neighbors
    in_activation_value = in_activation_value / total_accr_neighbors if total_accr_neighbors > 0 else 0

    return in_activation_value


def activation_function(activation_value, in_activation_value):
    """
    Function that calculates the new activation for an entity.

    Args:
        activation_value (float): The current activation value of the entity.
        in_activation_value (float): The input activation value of the entity.

    Returns:
        float: The new activation of the entity.
    """
    return activation_value + in_activation_value

