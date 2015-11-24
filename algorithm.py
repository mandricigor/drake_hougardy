

# __author__ = "IGOR MANDRIC"

import matplotlib.pyplot as plt
import networkx as nx
from copy import copy
import random


beta = 1


class Matching(dict):
    ############ CHECKED
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
    ########### CHECKED
    """
        Good beta-augmentation algorithm
    """
    possible_augmentations = []
    best_augm_one_edge = best_beta_augmentation_one_edge(graf, matching, edge, beta)
    possible_augmentations.append(best_augm_one_edge)
    best_augm_cycle = best_beta_augmentation_cycle(graf, matching, edge, beta)
    possible_augmentations.append(best_augm_cycle)
    e1, e2 = edge
    neighbors_e1 = [x for x in graf.edge[e1].keys() if x != e2]
    neighbors_e2 = [x for x in graf.edge[e2].keys() if x != e1]
    f = []
    for vertex in neighbors_e1:
        if not matching.has_edge(e1, vertex):
            f.append((e1, vertex))
    for vertex in neighbors_e2:
        if not matching.has_edge(e2, vertex):
            f.append((e2, vertex))
    ff = [x for x in f if win(x, edge, matching, graf) >= 0.5 * graf.edge[e1][e2]["weight"]]
    possible_augmentations.append(max_allowable(graf, matching, edge, f))
    possible_augmentations.append(max_allowable(graf, matching, edge, ff))
    #print possible_augmentations
    possible_augmentations = filter(None, possible_augmentations)
    return max(possible_augmentations, key=lambda x: x[0])


def best_beta_augmentation_one_edge(graf, matching, edge, beta):
    ############ CHECKED
    """
        The augmenting set is presented by just one edge
    """
    e1, e2 = edge
    e1_neighbors = [x for x in graf.edge[e1] if x != e2]
    e2_neighbors = [x for x in graf.edge[e2] if x != e1]
    neighbors = [(e1, x) for x in e1_neighbors]
    neighbors.extend([(e2, x) for x in e2_neighbors])
    augmentations = [(f, augmentation(f, edge, matching)) for f in neighbors]
    augmentations = [(f, augm) for f, augm in augmentations if augm]
    wins = [win2(f, augm, matching, graf, beta) for f, augm in augmentations]
    augms = [(w, [f], augm + [edge]) for w, (f, augm) in zip(wins, augmentations) if w > 0]
    if augms:
        return max(augms, key=lambda x: x[0])
    else:
        return None, None, None



def best_beta_augmentation_cycle(graf, matching, edge, beta):
    """
        Find best beta-augmentation with center `edge`
    """
    e1, e2 = edge
    augms = []
    e1_neighbors = [x for x in graf.edge[e1] if x != e2]
    e2_neighbors = [x for x in graf.edge[e2] if x != e1]

    marked_vertices = {}

    # check if M(e1) has an adjacent edge with e
    for vertex1 in e1_neighbors:
        a = (e1, vertex1)
        a_augm = augmentation(a, edge, matching)
        
        # for the case of cycle
        for u, v in a_augm:
            if u not in a:
                marked_vertices[u] = v
            elif v not in a:
                marked_vertices[v] = u

        win_a = win2(a, a_augm, matching, graf, beta_surplus=1)
        has_adjacent = False
        for a_ in a_augm:
            if adjacent(a_, edge, graf):
                has_adjacent = True
                break
        if has_adjacent:
            for vertex2 in e2_neighbors:
                b = (e2, vertex2)
                b_augm = augmentation(b, edge, matching)
                total_win = win_a + win2(b, b_augm, matching, graf, beta_surplus=1)
                if total_win > 0:
                    augms.append((total_win, [a, b], a_augm + b_augm + [edge]))

    # check if M(e2) has an adjacent edge with e
    for vertex2 in e2_neighbors:
        b = (e2, vertex2)

        if vertex2 in marked_vertices:
            cycle_win = graf.edge[e1][a_vertex]["weight"] + graf.edge[e2][vertex2]["weight"] - \
                graf.edge[a_vertex][vertex2]["weight"]
            a_vertex = marked_vertices[vertex2]
            if cycle_win > 0:
                augms.append((cycle_win, [b, (a_vertex, e1)], [edge, (vertex2, a_vertex)]))

        b_augm = augmentation(b, edge, matching)
        win_b = win2(b, b_augm, matching, graf, beta_surplus=1)
        has_adjacent = False
        for b_ in b_augm:
            if adjacent(b_, edge, graf):
                has_adjacent = True
                break
        if has_adjacent:
            for vertex1 in e1_neighbors:
                a = (e1, vertex1)
                a_augm = augmentation(a, edge, matching)
                total_win = win_b + win2(a, a_augm, matching, graf, beta_surplus=1)
                if total_win > 0:
                    augms.append((total_win, [a, b], a_augm + b_augm + [edge]))
    if augms:
        return max(augms, key=lambda x: x[0])
    else:
        return None, None, None


def augmentation(edge_a, edge_e, matching):
    ############# CHECKED
    """
        Returns an augmentation
    """
    a1, a2 = edge_a
    e1, e2 = edge_e

    def neighbors_(vertex, exclude_vertex):
        try:
            neighbors = [x for x in matching.edge[vertex].keys() if x != exclude_vertex]
        except KeyError:
            return []
        else:
            return [(vertex, x) for x in neighbors]

    neighbors = neighbors_(a1, a2) + neighbors_(a2, a1)
    augment = []
    for x, y in neighbors:
        if (x == e1 and y == e2) or (y == e1 and x == e2):
            continue
        else:
            augment.append((x, y))
    return augment


def bigger(el1, el2):
    ######### CHECKED
    """
        Default comparator
    """
    return el1[1] > el2[1]


def find_two_biggest(elements, comparator=bigger):
    ################# CHECKED
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
    ########## CHECKED
    """
        Check if f1 is adjacent with f2 in graf
    """
    x, y = edge_f1
    u, v = edge_f2
    are_adjacent = x == u or x == v or y == u or y == v
    return are_adjacent


def max_allowable(graf, matching, edge, edge_set_f):
    ############# CHECKED
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
    if matching.has_edge(x, y):
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
    #print f1, Fx, z
    #print "UNION:", x_union
    if not x_union:
        # Nothing to do!
        return None, None, None
    else:
        best = 0
        best_win = win(x_union[best], edge, matching, graf, verbose=False)
        for i in range(len(x_union[1:])):
            current_win = win(x_union[i + 1], edge, matching, graf, verbose=False)
            if current_win > best_win:
                best = i
        # now find the corresponding beta-augmentation
        # return now: best win, augmenting set (one edge), augmentation
        return best_win, [x_union[best]], augmentation(x_union[best], edge, matching) + [edge]


def win(edge_a, edge_e, matching, graf, beta_surplus=1.0, verbose=False):
    ######### CHECKED
    """
        Compute win* of an edge `a` relatively to
        The center* `e` of a beta-augmentation
        *win - how an edge improves your matching
        *center - the center of your augmentation
    """
    a1, a2 = edge_a
    e1, e2 = edge_e
    weight_a = graf.edge[a1][a2]["weight"]
    neighbors = augmentation(edge_a, edge_e, matching)
    if verbose:
        print neighbors
    neighbors_weight = sum([matching.edge[x][y]["weight"] for x, y in neighbors])
    return weight_a - beta_surplus * neighbors_weight


def win2(edge_a, neighbors, matching, graf, beta_surplus=1.0):
    ########### CHECKED
    """
        The same as `win` but takes a ready neighborhood
    """
    a1, a2 = edge_a
    weight_a = graf.edge[a1][a2]["weight"]
    neighbors_weight = sum([matching.edge[x][y]["weight"] for x, y in neighbors])
    return weight_a - beta_surplus * neighbors_weight

 
def maximal_matching(graf):
    ######## CHECKED
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
    ########## CHECKED
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
        maximal = make_maximal(matching, graf)
        matching_prime = nx.Graph()
        for x, y in maximal.get_iterator():
            matching_prime.add_edge(x, y, weight=graf.edge[x][y]["weight"])
    #draw_graph(graf, matching_prime)
    for edge in maximal.get_iterator():
        win, augmenting, augmented = good_beta_augmentation(graf, matching_prime, edge, beta)
        #print win, augmenting, augmented, edge
        if augmenting and augmented:
            # Wow! there exists a good beta-augmentation!
            # So, go ahead and improve your matching
            for x, y in augmented:
                if matching_prime.has_edge(x, y):
                    matching_prime.remove_edge(x, y)
            for x, y in augmenting:
                matching_prime.add_edge(x, y, weight=graf.edge[x][y]["weight"])
        #draw_graph(graf, matching_prime)
    return matching_prime 


def make_maximal(matching, graf):
    maximal = {}
    for x, y in matching.edges():
        maximal[x] = y
        maximal[y] = x
    for x, y in graf.edges():
        if x not in maximal and y not in maximal:
            maximal[x] = y
            maximal[y] = x
    return Matching(maximal)


def graph_to_matching(graf):
    ######## CHECKED
    match_dict = {}
    for x, y in graf.edges():
        match_dict[x] = y
        match_dict[y] = x
    return match_dict


def hougardy_matching(graf, iterations=1):
    ######### CHECKED
    """
        Implementation of Hougardy algorithm
        For Maximum Weight Matching
        With approximation ratio 2/3
    """
    matching_ = improve_matching(graf)
    for i in range(iterations - 1):
        matching_ = improve_matching(graf, matching_)
    matching_ = make_maximal(matching_, graf)
    return matching_


def weight_of_matching(graf, matching):
    ######## CHECKED
    weight = 0
    for x, y in matching.items():
        weight += graf.edge[x][y]["weight"]
    return weight / 2


def draw_graph(graf, matching):
    specific_edges = matching.edges()
    edge_colors = []
    edge_weights = []
    for x, y in graph.edges():
        if (x, y) in specific_edges:
            edge_colors.append('r')
            edge_weights.append(3)
        else:
            edge_colors.append('b')
            edge_weights.append(1)
    pos = nx.spring_layout(graph)
    nx.draw(graph, with_labels=True, edge_color=edge_colors, width=edge_weights, pos=pos)
    edge_labels = {}
    for x, y in graph.edges():
        edge_labels[(x, y)] = graph.edge[x][y]["weight"]
    nx.draw_networkx_edge_labels(graf,pos, font_size=10,font_family='sans-serif', edge_labels=edge_labels)
    plt.draw()
    plt.show()



if __name__ == "__main__":
    graph = nx.dodecahedral_graph()
    for x, y in graph.edges():
        graph.edge[x][y]["weight"] = random.randint(1, 100)
    super_matching = hougardy_matching(graph)
    exact_matching = nx.max_weight_matching(graph)
    print "weight of exact:", weight_of_matching(graph, exact_matching)
    print "ratio:", weight_of_matching(graph, super_matching) * 1.0 / weight_of_matching(graph, exact_matching)
    super_matching = hougardy_matching(graph, 2)
    print "ratio:", weight_of_matching(graph, super_matching) * 1.0 / weight_of_matching(graph, exact_matching)


