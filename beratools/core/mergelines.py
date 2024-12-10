import networkit as nk
from shapely.geometry import mapping
from shapely import Point, MultiLineString, LineString, reverse
from itertools import pairwise
from operator import itemgetter


class MergeLines:
    """ Merge line segments in MultiLineString """
    def __init__(self, multi_line):
        self.G = None
        self.line_segs = None
        self.multi_line = multi_line
        self.node_poly = None
        self.end = None

        self.create_graph()


    def create_graph(self):
        self.line_segs = list(self.multi_line.geoms)
        self.line_segs = [line for line in self.line_segs if line.length > 1e-3]  # TODO: check empty line and null geoms
        self.multi_line = MultiLineString(self.line_segs)
        m = mapping(self.multi_line)
        self.end = [(i[0], i[-1]) for i in m['coordinates']]

        self.G = nk.Graph(edgesIndexed=True)
        self.G.addNodes(2)
        self.G.addEdge(0, 1)

        self.node_poly = [Point(self.end[0][0]).buffer(1),  Point(self.end[0][1]).buffer(1)]
        
        for i, line in enumerate(self.end[1:]):
            node_exists = False
            pt = Point(line[0])
            pt_buffer = pt.buffer(1)
            
            for node in self.G.iterNodes():
                if self.node_poly[node].contains(pt):
                    node_exists = True
                    node_start = node
            if not node_exists:
                node_start = self.G.addNode()
                self.node_poly.append(pt_buffer)
            
            node_exists = False
            pt = Point(line[1])
            pt_buffer = pt.buffer(1)
            for node in self.G.iterNodes():
                if self.node_poly[node].contains(pt):
                    node_exists = True
                    node_end = node
            if not node_exists:
                node_end = self.G.addNode()
                self.node_poly.append(pt_buffer)

            edge = self.G.addEdge(node_start, node_end)

    def get_components(self):
        cc = nk.components.ConnectedComponents(self.G)
        cc.run()
        components = cc.getComponents()
        return components

    def is_single_path(self, component):
        single_path = True
        for node in component:
            neighbors = list(self.G.iterNeighbors(node))
            if len(neighbors) > 2:
                single_path = False

        return single_path
    def get_merged_line_for_component(self, component):
        sub = nk.graphtools.subgraphFromNodes(self.G, component)
        lines = None
        if nk.graphtools.maxDegree(sub) >= 3:  # not simple path
            edges = [self.G.edgeId(i[0], i[1]) for i in list(sub.iterEdges())]
            lines =  itemgetter(*edges)(self.line_segs)
        elif nk.graphtools.maxDegree(sub) == 2:
            lines = self.merge_single_line(component)

        return lines
    
    def find_path_for_component(self, component):
        neighbors = list(self.G.iterNeighbors(component[0]))
        path = [component[0]]
        right = neighbors[0]
        path.append(right)

        left = None
        if len(neighbors) == 2:
            left = neighbors[1]
            path.insert(0, left)

        neighbors = list(self.G.iterNeighbors(right))
        while len(neighbors) > 1:
            # print(neighbors)
            if neighbors[0] not in path:
                path.append(neighbors[0])
                right = neighbors[0]
            else:
                path.append(neighbors[1])
                right = neighbors[1]

            neighbors = list(self.G.iterNeighbors(right))

        # last node
        if neighbors[0] not in path:
            path.append(neighbors[0])

        # process left side
        if left:
            neighbors = list(self.G.iterNeighbors(left))
            while len(neighbors) > 1:
                if neighbors[0] not in path:
                    path.insert(0, neighbors[0])
                    left = neighbors[0]
                else:
                    path.insert(0, neighbors[1])
                    left = neighbors[1]

                neighbors = list(self.G.iterNeighbors(left))

            # last node
            if neighbors[0] not in path:
                path.insert(0, neighbors[0])

        return path
    
    def merge_single_line(self, component):
        path = self.find_path_for_component(component)

        print(path)
        pairs = list(pairwise(path))
        # print(pairs)

        line_list = [self.G.edgeId(i[0], i[1]) for i in pairs]
        # print(line_list)

        vertices = []

        for i, id in enumerate(line_list):
            pair = pairs[i]
            poly_t = self.node_poly[pair[0]]
            point_t = Point(self.end[id][0])
            if  poly_t.contains(point_t):
                line = self.line_segs[id]
            else:
                line = reverse(self.line_segs[id])

            vertices.extend(list(line.coords))
            last_vertex = vertices.pop()

        vertices.append(last_vertex)
        merged_line = LineString(vertices)

        return [merged_line]
    
    def merge_all_lines(self):
        components = self.get_components()
        lines = []
        for c in components:
            line = self.get_merged_line_for_component(c)
            if line:
                lines.extend(self.get_merged_line_for_component(c))
            else:  # TODO: check line
                print(line)

        if len(lines) > 1:
            return MultiLineString(lines)
        elif len(lines) == 1:
            return lines[0]
        else:
            return None