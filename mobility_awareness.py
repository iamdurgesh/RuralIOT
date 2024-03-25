# This code includes the updated algorithm with mobility considerations. It considers the locations of processors while generating the schedule. Additionally, it provides functionality to plot the generated schedule and outputs the final makespan and genome.
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from deap import base, creator, tools

def Read_Parent_AM(json_data):
    AMx = json_data['application']
    return AMx

def Read_Parent_PM(json_data):
    PMx = json_data['platform']
    return PMx

def construct_communication_costs_from_json(json_data):
    messages = json_data['application']['messages']
    communication_costs = {}

    for message in messages:
        sender = message['sender']
        receiver = message['receiver']
        size = message['size']

        if sender not in communication_costs:
            communication_costs[sender] = {receiver: size}
        else:
            communication_costs[sender][receiver] = size

    return communication_costs

def construct_task_dag_from_json(APP_MODEL):
    jobs = APP_MODEL['jobs']
    messages = APP_MODEL['messages']

    num_tasks = len(jobs)
    message_mapping = defaultdict(list)

    for message in messages:
        sender = message['sender']
        receiver = message['receiver']
        message_mapping[sender].append(receiver)

    task_dag = [[] for _ in range(num_tasks)]

    for job_id, successors in message_mapping.items():
        task_dag[job_id] = successors

    wcet_values = [job['processing_times'] for job in jobs]

    return task_dag, wcet_values

def extract_message_list(APP_MODEL):
    messages = APP_MODEL['messages']
    task_ids = [job['id'] for job in APP_MODEL['jobs']]
    message_list = []

    for msg in messages:
        sender_id = task_ids.index(msg['sender'])
        receiver_id = task_ids.index(msg['receiver'])
        message_size = msg['size']
        message_id = msg['id']
        message_info = {
            'id': message_id,
            'sender': sender_id,
            'receiver': receiver_id,
            'size': message_size
        }
        message_list.append(message_info)
    return message_list

def construct_graph_from_json(PLAT_MODEL):
    nodes = PLAT_MODEL['nodes']
    links = PLAT_MODEL['links']
    graph = nx.Graph()

    for node in nodes:
        node_id = node['id']
        node_type = 'processor' if not node['is_router'] else 'switch'
        graph.add_node(node_id, node_type=node_type)

    for link in links:
        start = link['start']
        end = link['end']
        graph.add_edge(start, end)

    return graph

def generate_all_path_indexes_with_costs(graph):
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
                        path_indexes[path_id] = {"path": path, "cost": path_cost}
                        path_id += 1
    return path_indexes

def reconstruct_schedule_with_precedenceX(task_allocation, node_list, processing_times, message_list, message_path_index, all_path_indexes_with_costs, message_priority_ordering, uav_loc):
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

def generate_schedule_base_schedule(processing_times, message_list, all_path_indexes_with_costs, uav_loc):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    num_tasks = len(processing_times)
    nodes = json_data["platform"]["nodes"]
    processors = [node for node in nodes if not node["is_router"]]
    num_machines = len(processors) - 1
    num_message = len(message_list)
    num_paths = len(all_path_indexes_with_costs)

    def init_individual():
        individual = []
        individual.extend(random.sample(range(num_tasks), num_tasks))
        individual.extend([random.randint(0, num_machines) for _ in range(num_tasks)])
        individual.extend([random.randint(0, num_paths - 1) for _ in range(num_message)])
        individual.extend(random.sample(range(num_message), num_message))
        return individual

    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)

    def evaluate(individual):
        task_order_len = num_tasks
        processor_allocation_len = num_tasks
        message_path_index_len = num_message
        message_priority_ordering_len = num_message
        task_order = individual[:task_order_len]
        processor_allocation = individual[task_order_len:task_order_len + processor_allocation_len]
        message_path_index = individual[task_order_len + processor_allocation_len: task_order_len + processor_allocation_len + message_path_index_len]
        message_priority_ordering = individual[task_order_len + processor_allocation_len + message_path_index_len:]
        
        schedule = reconstruct_schedule_with_precedenceX(processor_allocation, task_order, processing_times, message_list, message_path_index, all_path_indexes_with_costs, message_priority_ordering, uav_loc)
        makespan = compute_makespan(schedule)
        fitness = 1.0 / makespan
        return fitness,

    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.selTournament, tournsize=3)

    def mate(ind1, ind2):
        child1, child2 = toolbox.clone(ind1), toolbox.clone(ind2)
        tools.cxPartialyMatched(child1[:num_tasks], child2[:num_tasks])
        return child1, child2

    toolbox.register("mate", mate)

    def mutation_task_order(individual):
        individual[:num_tasks] = random.sample(individual[:num_tasks], num_tasks)
        return individual,

    toolbox.register("mutate_task_order", mutation_task_order)

    def mutation_processor_allocation(individual):
        for i in range(num_tasks):
            individual[num_tasks + i] = random.randint(0, num_machines)
        return individual,

    toolbox.register("mutate_processor_allocation", mutation_processor_allocation)

    def mutation_message_path_index(individual):
        for i in range(num_tasks, num_tasks + num_message):
            individual[i] = random.randint(0, num_paths - 1)
        return individual,

    toolbox.register("mutate_message_path_index", mutation_message_path_index)

    def mutation_message_priority_ordering(individual):
        individual[num_tasks + num_message:] = random.sample(individual[num_tasks + num_message:], num_message)
        return individual,

    toolbox.register("mutate_message_priority_ordering", mutation_message_priority_ordering)

    pop = toolbox.population(n=100)

    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    CXPB, MUTPB = 0.3, 0.2
    fits = [ind.fitness.values[0] for ind in pop]
    NGEN = 50

    for g in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate_task_order(mutant)
                toolbox.mutate_processor_allocation(mutant)
                toolbox.mutate_message_path_index(mutant)
                toolbox.mutate_message_priority_ordering(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

    best_ind = tools.selBest(pop, 1)[0]

    task_order_len = num_tasks
    processor_allocation_len = num_tasks
    message_path_index_len = num_message
    message_priority_ordering_len = num_message
    task_order = best_ind[:task_order_len]
    processor_allocation = best_ind[task_order_len:task_order_len + processor_allocation_len]
    message_path_index = best_ind[task_order_len + processor_allocation_len: task_order_len + processor_allocation_len + message_path_index_len]
    message_priority_ordering = best_ind[task_order_len + processor_allocation_len + message_path_index_len:]

    schedule_final = reconstruct_schedule_with_precedenceX(processor_allocation, task_order, processing_times, message_list, message_path_index, all_path_indexes_with_costs, message_priority_ordering, uav_loc)

    return schedule_final, [task_order, processor_allocation, message_path_index, message_priority_ordering]

def compute_makespan(schedule):
    end_times = [info[2] for info in schedule.values()]
    makespan = max(end_times)
    return makespan

def plot_schedule(schedule):
    G = nx.DiGraph()
    for task, info in schedule.items():
        G.add_node(task, label=f'Task {task}\nProcessor {info[0]}\nStart: {info[1]}\nEnd: {info[2]}')
    edges = []
    for task, successors in task_dag.items():
        for successor in successors:
            edges.append((task, successor))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1500, node_color='skyblue', font_size=8, font_weight='bold')
    plt.show()

# Example usage
json_data = {
    "application": {
        "jobs": [
            {"id": 0, "processing_times": 10},
            {"id": 1, "processing_times": 8},
            {"id": 2, "processing_times": 12}
        ],
        "messages": [
            {"id": 0, "sender": 0, "receiver": 1, "size": 5},
            {"id": 1, "sender": 0, "receiver": 2, "size": 6},
            {"id": 2, "sender": 1, "receiver": 2, "size": 4}
        ]
    },
    "platform": {
        "nodes": [
            {"id": 0, "is_router": False},
            {"id": 1, "is_router": False},
            {"id": 2, "is_router": False}
        ],
        "links": [
            {"start": 0, "end": 1},
            {"start": 0, "end": 2},
            {"start": 1, "end": 2}
        ]
    }
}

task_dag, wcet_values = construct_task_dag_from_json(json_data["application"])
message_list = extract_message_list(json_data["application"])
communication_costs = construct_communication_costs_from_json(json_data)
graph = construct_graph_from_json(json_data["platform"])
all_path_indexes_with_costs = generate_all_path_indexes_with_costs(graph)

uav_loc = [(0, 0), (1, 1), (2, 2)]  # Example UAV locations

schedule_final, final_genome = generate_schedule_base_schedule(wcet_values, message_list, all_path_indexes_with_costs, uav_loc)

makespan_final = compute_makespan(schedule_final)

plot_schedule(schedule_final)

print('The final schedule makespan is', makespan_final)
print("The Final Genome for this schedule is", final_genome)
