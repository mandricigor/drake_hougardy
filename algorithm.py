

# __author__ = "IGOR MANDRIC"

import networkx as nx
from copy import copy
import random


beta = 2


class Matching(dict):
    """
        Class for matching in graf
    """
    def __init__(self, *args, **kw):
        super(Matching, self).__init__(*args, **kw)

    def get_iterator(self):
        iterator = set()
        for x, y in self.items():
            if x > y:
                x, y = y, x
            iterator.add((x, y))
        return list(iterator)


def good_beta_augmentation(graf, matching, edge, beta):
    """
        Good beta-augmentation algorithm
    """
    possible_augmentations = []
    best_augm = best_beta_augmentation(graf, matching, edge, beta)
    possible_augmentations.append(best_augm)
    e1, e2 = edge
    neighbors_e1 = graf.edge[e1].keys()
    neighbors_e2 = graf.edge[e2].keys()
    f = []
    for vertex in neighbors_e1:
        if not matching.has_edge(e1, vertex):
            f.append((e1, vertex))
    for vertex in neighbors_e2:
        if not matching.has_edge(e2, vertex):
            f.append((e2, vertex))
    ff = [x for x in f if win(edge, x, matching, graf) > 0.5 * matching.edge[e1][e2]["weight"]]
    possible_augmentations.append(max_allowable(graf, matching, edge, f))
    possible_augmentations.append(max_allowable(graf, matching, edge, ff))
    possible_augmentations = filter(None, possible_augmentations)
    return max(possible_augmentations, key=lambda x: x[0])


def best_beta_augmentation(graf, matching, edge, beta):
    """
        Find best beta-augmentation with center `edge`
    """
    e1, e2 = edge
    e1_neighbors = [x for x in graf.edge[e1] if x != e2]
    e2_neighbors = [x for x in graf.edge[e2] if x != e1]
    neighbors = [(e1, x) for x in e1_neighbors]
    neighbors.extend([(e2, x) for x in e2_neighbors])
    augmentations = [(f, augmentation(f, edge, matching)) for f in neighbors]
    augmentations = [(f, augm) for f, augm in augmentations if augm]
    wins = [win2(f, augm, matching, graf, beta) for f, augm in augmentations]
    augms = [(w, augm) for w, (f, augm) in zip(wins, augmentations) if w > 0]
    if augms:
        return max(augms, key=lambda x: x[0])
    else:
        return None, None


def augmentation(edge_a, edge_e, matching):
    """
        Returns an augmentation
    """
    a1, a2 = edge_a
    e1, e2 = edge_e

    def neighbors_(vertex):
        try:
            neighbors = matching.edge[vertex].keys()
        except KeyError:
            return []
        else:
            return [(vertex, x) for x in neighbors if x not in edge_e]

    return neighbors_(a1) + neighbors_(a2)


def bigger(el1, el2):
    """
        Default comparator
    """
    return el1[1] > el2[1]


def find_two_biggest(elements, comparator=bigger):
    """
        Given a list of elements and a comparator
        Return two biggest elements of the list
    """
    if len(elements) == 0:
        return None, None
    elif len(elements) == 1:
        return elements[0][0], elements[0][0]
    else:
        if comparator(elements[0], elements[1]):
            big1, big2 = elements[:2]
        else:
            big1, big2 = elements[1], elements[0]
        for el in elements[2:]:
            if comparator(el, big1):
                big2 = big1
                big1 = el
            elif comparator(el, big2):
                big2 = el
        return big1[0], big2[0]


def adjacent(edge_f1, edge_f2, graf):
    """
        Check if f1 is adjacent with f2 in graf
    """
    x, y = edge_f1
    u, v = edge_f2
    are_adjacent = (x, u) in graf.edges() or (x, v) in graf.edges() or \
        (u, y) in graf.edges() or (v, y) in graf.edges()
    return are_adjacent


def max_allowable(graf, matching, edge, edge_set_f):
    """
        Max allowable algorithm
    """
    x, y = edge
    Fx = [f for f in edge_set_f if x in f]
    Fy = [f for f in edge_set_f if y in f]
    surpluses_x = [(f, win(f, edge, matching, graf, beta)) for f in Fx]
    surpluses_y = [(f, win(f, edge, matching, graf, beta)) for f in Fy]
    f1, f2 = find_two_biggest(surpluses_x)
    f3, f4 = find_two_biggest(surpluses_y)
    if y in matching.edge[x]:
        z = matching.edge[x][y]["weight"]
    else:
        z = 0

    def good(edge_f, edge_f1, edge, matching, value_z, beta):
        win1 = win(edge_f, edge, matching, graf, beta)
        win2 = win(edge_f1, edge, matching, graf, beta)
        return win1 + win2 >= beta * value_z

    x1 = [f for f in Fx if not adjacent(f, f1, graf) and good(f, f1, edge, matching, z, beta)]
    x2 = [f for f in Fx if not adjacent(f, f2, graf) and good(f, f2, edge, matching, z, beta)]
    x3 = [f for f in Fy if not adjacent(f, f3, graf) and good(f, f3, edge, matching, z, beta)]
    x4 = [f for f in Fy if not adjacent(f, f4, graf) and good(f, f4, edge, matching, z, beta)]
    x_union = x1 + x2 + x3 + x4
    if not x_union:
        # Nothing to do!
        return None, None
    else:
        best = 0
        best_win = win(x_union[best], edge, matching, graf, beta)
        for i in range(len(x_union[1:])):
            current_win = win(x_union[i + 1], edge, matching, graf, beta)
            if current_win > best_win:
                best = i
        # now find the corresponding beta-augmentation 
        return best_win, augmentation(edge, x_union[best], matching) 


def win(edge_a, edge_e, matching, graf, beta_surplus=1.0):
    """
        Compute win* of an edge `a` relatively to
            the center* `e` of a beta-augmentation
        *win - how an edge improves your matching
        *center - the center of your augmentation
    """
    a1, a2 = edge_a
    e1, e2 = edge_e
    weight_a = graf.edge[a1][a2]["weight"]
    neighbors = augmentation(edge_a, edge_e, matching)
    neighbors_weight = sum([matching.edge[x][y]["weight"] for x, y in neighbors])
    return weight_a - beta_surplus * neighbors_weight


def win2(edge_a, neighbors, matching, graf, beta_surplus=1.0):
    """
        The same as `win` but takes a ready neighborhood
    """
    a1, a2 = edge_a
    weight_a = graf.edge[a1][a2]["weight"]
    neighbors_weight = sum([matching.edge[x][y]["weight"] for x, y in neighbors])
    return weight_a - beta_surplus * neighbors_weight

 
def maximal_matching(graf):
    """
        Given a graph `graf`
        Return its Maximum Weight Matching
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
    return matching


def improve_matching(graf, matching=None):
    """
        1. Make matching M maximal
        2. For each edge in it, try to make a beta-augmentation
    """
    if not matching:
        maximal = maximal_matching(graf)
        matching_prime = nx.Graph()
        for x, y in maximal.get_iterator():
            matching_prime.add_edge(x, y, weight=graf.edge[x][y]["weight"])
    else:
        matching_prime = copy(matching)
        maximal = Matching(graph_to_matching(copy(matching)))
    for edge in maximal.get_iterator():
        win, augm = good_beta_augmentation(graf, matching_prime, edge, beta)
        if win and augm:
            # Wow! there exists a good beta-augmentation!
            # So, go ahead and improve your matching
            matching_prime.remove_edge(*edge)
            for x, y in augm:
                matching_prime.add_edge(x, y, weight=graf.edge[x][y]["weight"])
    return matching_prime 


def graph_to_matching(graf):
    match_dict = {}
    for x, y in graf.edges():
        match_dict[x] = y
        match_dict[y] = x
    return match_dict



def hougardy_matching(graf, iterations=1):
    """
        Implementation of Hougardy algorithm
        For Maximum Weight Matching
        With approximation ratio 2/3
    """
    matching_ = improve_matching(graf)
    for i in range(iterations - 1):
        matching_ = improve_matching(graf, matching_)
    return graph_to_matching(matching_)


def weight_of_matching(graf, matching):
    weight = 0
    for x, y in matching.items():
        weight += graf.edge[x][y]["weight"]
    return weight / 2


if __name__ == "__main__":
    graph = nx.dense_gnm_random_graph(50, 2300)
    for x, y in graph.edges():
        graph.edge[x][y]["weight"] = random.randint(1, 100)
    super_matching = hougardy_matching(graph)
    exact_matching = nx.max_weight_matching(graph)
    print "ratio:", weight_of_matching(graph, super_matching) * 1.0 / weight_of_matching(graph, exact_matching)
    print len(super_matching), len(exact_matching)
    #super_matching = hougardy_matching(graph, 10)
    #print "ratio:", weight_of_matching(graph, super_matching) * 1.0 / weight_of_matching(graph, exact_matching)

