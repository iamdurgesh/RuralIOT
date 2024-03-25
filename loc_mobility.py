def generate_all_path_indexes_with_costs(graph, location_coordinates):
    processors = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'processor']
    switches = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'switch']

    def find_all_paths(source, target, path=[]):
        path = path + [source]
        if source == target:
            return [path]
        if source not in graph:
            return []
        paths = []
        for node in graph[source]:
            if node not in path:
                newpaths = find_all_paths(node, target, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    path_indexes = {}
    path_id = 0
    for source in processors:
        for target in processors:
            if source != target:
                all_paths = find_all_paths(source, target)
                all_paths = [path for path in all_paths if any(node in path for node in switches)]
                if all_paths:
                    for path in all_paths:
                        path_cost = len(path) - 1
                        # Calculate the mobility factor based on location coordinates
                        mobility_factor = 0
                        for i in range(len(path) - 1):
                            node1 = path[i]
                            node2 = path[i+1]
                            if node1 in location_coordinates and node2 in location_coordinates:
                                x1, y1 = location_coordinates[node1]
                                x2, y2 = location_coordinates[node2]
                                mobility_factor += abs(x2 - x1) + abs(y2 - y1)
                        # Add the path, its ID, and its cost to the result
                        path_indexes[path_id] = {"path": path, "cost": path_cost, "mobility_factor": mobility_factor}
                        path_id += 1
    return path_indexes