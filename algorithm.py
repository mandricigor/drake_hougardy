

# __author__ = "IGOR MANDRIC"

import networkx as nx
from copy import copy



beta = 1.4



def adjacent(f, f1, graf):
    x, y = f
    u, v = f1
    are_adjacent = (x, u) in graf.edges() or (x, v) in graf.edges() or \
        (u, y) in graf.edges() or (v, y) in graf.edges()
    return are_adjacent


def good(edge_f, edge_f1, edge, matching, value_z, beta):
    return win(edge_f, edge, matching, beta) + win(edge_f1, edge, matching, beta) >= beta * value_z



def max_allowable(graf, matching, edge, edge_set_f):
    x, y = edge
    Fx = [f for f in edge_set_f if x in f]
    Fy = [f for f in edge_set_f if y in f]
    surpluses_x = [(f, win(f, edge, matching, beta)) for f in Fx]
    surpluses_y = [(f, win(f, edge, matching, beta)) for f in Fy]
    f1, f2 = find_two_biggest(surpluses_x)
    f3, f4 = find_two_biggest(surpluses_y)
    if y in matching.edge[x]:
        # if e \in M
        z = matching.edge[x][y]["weight"]
    else:
        z = 0
    x1 = [f for f in Fx if not adjacent(f, f1, graf) and good(f, f1, edge, matching, z, beta)]
    x2 = [f for f in Fx if not adjacent(f, f2, graf) and good(f, f2, edge, matching, z, beta)]
    x3 = [f for f in Fx if not adjacent(f, f3, graf) and good(f, f3, edge, matching, z, beta)]
    x4 = [f for f in Fx if not adjacent(f, f4, graf) and good(f, f4, edge, matching, z, beta)]
    return 0



def good_beta_augmentation(graf, matching, edge):
    e1, e2 = edge
    # find best beta-augmentation with center `edge`
    # that contains at most one edge not in `matching`
    a1 = 0
    # find best beta-augmentation with center `edge`
    # that contains a cycle
    a2 = 0
    # find a3
    f = [(e1, graf.edge[e1][x]) for x in graf.edge[e1].keys() if not matching.edge[x] and win((e1, graf.edge[e1][x]), edge, matching) > 0.5 * graf.edge[e1][e2]["weight"]] + \
        [(e2, graf.edge[e2][x]) for x in graf.edge[e2].keys() if not matching.edge[x] and win((e2, graf.edge[e2][x]), edge, matching) > 0.5 * graf.edge[e1][e2]["weight"]]
    a3 = max_allowable(graf, matching, edge, f)
    # find a4
    f = 0
    print a1, a2, a3
    a4 = max_allowable(graf, matching, edge, f)
    return max(a1, a2, a3, a4)



def win(edge_a, edge_e, matching, beta_surplus = 1.0):
    a1, a2 = edge_a
    e1, e2 = edge_e
    weight_a = matching.edge[a1][a2]["weight"]
    neighbors_weight = sum([matching.edge[a1][x]["weight"] for x in matching.edge[a1].keys() if x not in edge_e]) + \
        sum([matching.edge[a2][x]["weight"] for x in matching.edge[a2].keys() if x not in edge_e])
    return weight_a - beta_surplus * neighbors_weight
    
    




def best_beta_augmentation(graf, matching, edge):
    x_prime, x_x_prime_weight = None, 0
    y_prime, y_y_prime_weight = None, 0
    x_vicinity = matching.edge[x].keys()
    if x_vicinity:
        x_prime = x_vicinity[0]
        x_x_prime_weight = matching.edge[x][x_prime]["weight"]
    y_vicinity = matching.edge[y].keys()
    if y_vicinity:
        y_prime = y_vicinity[0]
        y_y_prime_weight = matching.edge[y][y_prime]["weight"]
    if x_prime and y_prime and (x == y_prime and y == x_prime):
        current_weight = x_x_prime_weight
    else:
        current_weight = x_x_prime_weight + y_y_prime_weight
    # try to find a beta-augmentation
    x_neighbors = [(u, matching.edge[x][u]["weight"]) for u in matching.edge[x].keys() if u != x_prime]
    y_neighbors = [(u, matching.edge[y][u]["weight"]) for u in matching.edge[y].keys() if u != y_prime]
    x_prime2 = None
    if x_neighbors:
        x_prime2 = max(x_neighbors, key=lambda u: u[1])
    y_prime2 = None
    if y_neighbors:
        y_prime2 = max(y_neighbors, key=lambda u: u[1])
    if x_prime2 and y_prime2 and x_prime2[0] != y_prime2[0]:
        new_weight = x_prime2[1] + y_prime2[1]
    elif x_prime2 and y_prime2 and x_prime2[0] == y_prime2[1]:
        new_weight = x_prime2[1]
    elif x_prime2 and not y_prime2:
        new_weight = x_prime2[1]
    elif not x_prime2 and y_prime2:
        new_weight = y_prime2[1]
    else:
        new_weight = 0
    if new_weight > beta * current_weight:
        return x, y, x_prime, y_prime, x_prime2, y_prime2
    else:
        return None 




class Matching(dict):
    def __init__(self, *args, **kw):
        super(Matching, self).__init__(*args, **kw)

    def get_iterator(self):
        iterator = set()
        for x, y in self.items():
            if x > y:
                x, y = y, x
            iterator.add((x, y))
        return list(iterator)



class Augmentation(object):
    def __init__(self, remove_edges=None, add_edges=None, gain=0):
        self.remove_edges = remove_edges
        self.add_edges = add_edges
        self.gain = gain

    



def hougardy_matching(graf):
    """ Given a graph `graf`,
        return its Maximum Weight Matching
    """
    # make a maximal matching
    matching = Matching()
    for x, y in graf.edges():
        a, b = matching.get(x), matching.get(y)
        if a == None and b == None:
            # if the nodes are not involved in any matchings,
            # add them to the matching
            matching[x] = y
            matching[y] = x
    matching_prime = nx.Graph()
    # create a graph for matching for an easier implementation
    for x, y in matching.get_iterator():
        matching_prime.add_edge(x, y, weight=graf.edge[x][y]["weight"])

    for edge in matching.get_iterator():
        # find the weight of current local matching
        beta_augmentation = good_beta_augmentation(graf, matching_prime, edge)
    return matching_prime




if __name__ == "__main__":
    graph = nx.Graph()
    numNodes = 6

    #Add vertices
    for node1 in range(numNodes):
        graph.add_node(node1)

    #Add edges - complete graph
    for node1 in range(numNodes):
        for node2 in range(node1+1, numNodes):
            graph.add_edge(node1, node2, weight=(node1+node2)*2)
    print graph.edges()
    print graph.nodes()
    print hougardy_matching(graph)


