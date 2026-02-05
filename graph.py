import networkx as nx
import csv


# 创建边，类型为列表，[h,t,dict]，dict的key计划采用relation及其嵌入emb
def build_edges(e_dict, r_dict, dataset="DBpedia"):
    edges = []
    if dataset == "YAGO4":
        with open('../data/YAGO4/raw/yago-wd-facts.nt') as f:
            for line in f:
                line = line.split('\t')[:3]
                edge = [e_dict[line[0]], e_dict[line[2]], {'relation': line[1], 'r_dict': r_dict[line[1]]}]
                edges.append(edge)
    else:
        with open('../data/DBpedia/raw/mappingbased_objects_wkd_uris_en.ttl') as f:
            for line in f:
                line = line.split(' ')[0:3]
                if 'type' not in line[1] and 'homepage' not in line[1] and line[0] != '#' \
                        and line[0] in e_dict and line[2] in e_dict:
                    edge = [e_dict[line[0]], e_dict[line[2]], {'relation': line[1], 'r_dict': r_dict[line[1]]}]
                    edges.append(edge)
    return edges


# TODO:统一函数，完整图和部分图的边都使用该接口创建
def build_edges_incomplete(e_dict, r_dict, dataset="DBpedia"):
    edges = []
    file_path = '../data/DBpedia/mid/kg_data_train.csv'
    if dataset == "YAGO4":
        file_path = '../data/YAGO4/mid/kg_data_train.csv'
    with open(file_path) as csvfile:
        f = csv.reader(csvfile)
        next(f)
        for line in f:
            edge = [e_dict[line[1]], e_dict[line[3]], {'relation': line[2], 'r_dict': r_dict[line[2]]}]
            edges.append(edge)
    return edges


# 创建节点字典，包含属性和本体，key为实体类似Q100001，value为数量和属性，[('1',{})]
def build_nodes(e_dict, dataset="DBpedia"):
    nodes = []
    file_path = '../data/DBpedia/mid/is_data_train.csv'
    if dataset == "YAGO4":
        file_path = '../data/DBpedia/mid/is_data_train.csv'
    else:
        with open('../data/DBpedia/raw/specific_mappingbased_properties_wkd_uris_en.ttl') as f:  # 属性
            for line in f:
                line = line.split(' ')[0:3]
                if 'type' not in line[1] and 'homepage' not in line[1] and line[0] != '#' and line[0] in e_dict:
                    prop = {line[1]: line[2]}
                    nodes.append((e_dict[line[0]], prop))
    with open(file_path) as csvfile:  # 本体
        f = csv.reader(csvfile)
        next(f)
        line = f.__next__()
        e = line[1]
        ot = [line[2]]
        for line in f:
            if line[1] != e:
                if e in e_dict and len(ot) > 0:
                    nodes.append((e_dict[e], {"ontology": ot}))
                e = line[1]
                ot = []
            ot.append(line[2])
    return nodes


def build_graph(e_prop, edges):
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    graph.add_nodes_from(e_prop)
    return graph


def valid_exist_path(graph, query_type, start, end):
    if query_type == '2p':
        pass


# n  len pos
# 1p: 1, pos[6]
# 2p: 1, pos[5], 2, pos[6]
# 3p: 1, pos[4], 2, pos[5], 3, pos[6]
def find_all_paths(graph, n, pos):
    end = pos[7]
    start = pos[6 - n]
    paths = []
    visited = set()

    def dfs(node, path):
        if len(path) == n + 1:
            if node == end:
                paths.append(path)
            return

        visited.add(node)
        mid = pos[6 - n + len(path)]
        neighbors = [n for n in graph.neighbors(node) if graph[node][n]['r_dict'] == mid]
        for neighbor in neighbors:
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
        visited.remove(node)

    dfs(start, [start])
    return paths


# 构建子图并计算相似度
def build_subgraph(graph, start, depth):
    sub_edges = nx.bfs_edges(graph, start, depth_limit=depth)
    subgraph = graph.edge_subgraph(sub_edges)
    return subgraph


def get_subgraph_nodes(graph, node, query_type, only_leaf):
    subgraph_nodes = set()
    visited = set()

    def dfs(node, path):
        if only_leaf:
            if query_type == '1p' and len(path) == 2:
                subgraph_nodes.add(node)
            if query_type == '2p' and len(path) == 3:
                subgraph_nodes.add(node)
            if query_type == '3p' and len(path) == 4:
                subgraph_nodes.add(node)
        else:
            subgraph_nodes.add(node)

        if len(path) == 4:
            return

        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
        visited.remove(node)

    dfs(node, [node])
    return subgraph_nodes
