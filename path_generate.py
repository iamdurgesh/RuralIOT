def generate_all_path_indexes_with_costs(graph):                           # Passing the PM as an argument , geneartes all paths between processors in the graph while calcualting the cost
    processors = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'processor']
    # creating a list with the processor nodes ids
    switches = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'switch']
    # creating a list with the switche nodes ids

    def find_all_paths(source, target, path=[]): # A recursive function to find all paths between a give source node and target node, source and target are the iterables from the below for loop

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

    path_indexes = {}                             # Initialzing a dict. to store results (paths and costs)
    path_id = 0
    for source in processors:                     # iterating through the processors list in the PM [0,1,2]
        for target in processors:                 # iterating through processors list in PM [0,1,2], done to consider all pairs between source and target nodes
            if source != target:                  # checking if source and target nodes are not the same as we don't need a path from the processor to itself
                all_paths = find_all_paths(source, target)
                all_paths = [path for path in all_paths if any(node in path for node in switches)] # filtering the paths to keep the ones with only one switch node
                if all_paths:
                    for path in all_paths:
                        # Compute the cost as the number of edges in the path
                        path_cost = len(path) - 1                                   # Computing the cost by subtracting 1 from the number of nodes in the path.
                        # Add the path, its ID, and its cost to the result
                        path_indexes[path_id] = {"path": path, "cost": path_cost}
                        path_id += 1
    return path_indexes