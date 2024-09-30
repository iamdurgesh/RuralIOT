import networkx as nx
import json
import random
from copy import deepcopy

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


# -------------------------Scheduling Reconstruction Algorithm ---------------------------------------------------------------

def reconstruct_schedule_with_mobility(task_allocation, node_list, processing_times, message_list, message_path_index, all_path_indexes_with_costs, message_priority_ordering):
    message_list_copy = deepcopy(message_list)
    num_processors = max(task_allocation) + 1
    processors = [[] for _ in range(num_processors)]
    for task, processor in enumerate(task_allocation):
        processors[processor].append(task)

    schedule = {}
    task_completion_times = [0] * len(node_list)
    message_dict = defaultdict(list)
    for i in range(len(node_list)):
        message_dict[i] = []

    message_priority_dict = {message_id: priority for priority, message_id in enumerate(message_priority_ordering)}

    for idx, message in enumerate(message_list_copy):
        path_id = message_path_index[idx]
        path = all_path_indexes_with_costs[path_id]["path"]
        path_cost = all_path_indexes_with_costs[path_id]["cost"]
        mobility_factor = all_path_indexes_with_costs[path_id]["mobility_factor"]

        message["size"] += path_cost

        message_dict[message["receiver"]].append((message["sender"], message["size"], message_priority_dict[message["id"]]))

    for receiver, messages in message_dict.items():
        message_dict[receiver] = sorted(messages, key=lambda x: x[2])

    current_time_per_processor = [0] * num_processors
    completed_tasks = set()
    ready_tasks = set(range(len(node_list)))

    while ready_tasks:
        task = ready_tasks.pop()
        task_id = node_list[task]
        processor = task_allocation[task]
        i = processor

        predecessors = message_dict[task_id]
        if all(p in completed_tasks for p, _, _ in predecessors):
            if predecessors:
                start_time = max(current_time_per_processor[i], max(task_completion_times[sender] + size for sender, size, _ in predecessors))
            else:
                start_time = current_time_per_processor[i]

            end_time = start_time + processing_times[task_id]
            schedule[task_id] = (i, start_time, end_time)
            task_completion_times[task_id] = end_time

            current_time_per_processor[i] = end_time
            completed_tasks.add(task_id)
        else:
            ready_tasks.add(task)

    return schedule



#------------------------  Updating Fitness calculation based on the mobility factor in the base schedule

def evaluate(individual, processing_times, message_list, all_path_indexes_with_costs):
    task_order_len = num_tasks
    processor_allocation_len = num_tasks
    message_path_index_len = num_message
    message_priority_ordering_len = num_message
    
    task_order = individual[:task_order_len]
    processor_allocation = individual[task_order_len:task_order_len + processor_allocation_len]
    message_path_index = individual[task_order_len + processor_allocation_len : task_order_len + processor_allocation_len + message_path_index_len]
    message_priority_ordering = individual[task_order_len + processor_allocation_len + message_path_index_len :]
    
    # Reconstruct schedule with mobility
    schedule = reconstruct_schedule_with_mobility(task_order, processors, processing_times, message_list, message_path_index, all_path_indexes_with_costs, message_priority_ordering)
    
    # Compute makespan considering mobility
    makespan = compute_makespan(schedule)
    
    # Fitness calculation with mobility factor
    total_time = makespan + mobility_factor_adjustment(schedule, all_path_indexes_with_costs)
    fitness = 1.0 / total_time
    
    return fitness,



#Mobility parameter into the individual representation and genetic operators:
"""
1.	Initialization: extending the create_individual() function to include the mobility factor.
2.	Crossover: modifying the crossover function to handle the mobility factor if needed.
3.	Mutation: updating the mutation functions to ensure that mobility factors are appropriately modified."""

# Initialization
def create_individual():
    individual = []
    individual.extend(toolbox.task_order())
    individual.extend(toolbox.processor_allocation())
    individual.extend(toolbox.message_path_index())
    individual.extend(toolbox.message_priority_ordering())
    individual.append(random.uniform(0.1, 1.0))  # Adding a random mobility factor
    return individual

# Crossover
def mate(ind1, ind2, task_order_len):
    child1, child2 = toolbox.clone(ind1), toolbox.clone(ind2)

    # Perform crossover on task_order, processor_allocation, message_path_index, and message_priority_ordering

    # Crossover for mobility factor (if needed)
    # Assuming the mobility factor is represented as the last element in the individual
    child1_mobility = ind1[-1]
    child2_mobility = ind2[-1]
    if random.random() < 0.5:  # 50% chance of swapping mobility factor
        child1[-1] = child2_mobility
        child2[-1] = child1_mobility

    del child1.fitness.values
    del child2.fitness.values

    return child1, child2

# Mutation
def mutation_mobility_factor(individual):
    # Mutation for the mobility factor
    # Assuming the mobility factor is represented as the last element in the individual
    mobility_factor = individual[-1]
    mutated_mobility_factor = max(0.1, min(1.0, mobility_factor + random.uniform(-0.1, 0.1)))  # Mutate within the range [0.1, 1.0]
    individual[-1] = mutated_mobility_factor
    return individual,

# Register mutation operators
toolbox.register("mutate_mobility_factor", mutation_mobility_factor)


# register mutation operators
toolbox.receiver("mutate_mobilit_factor" ,mutation_mobility_factor)
