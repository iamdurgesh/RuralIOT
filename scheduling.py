from deap import creator, base, tools
import random

# -------Final Scheduling Algorithm for reconstruction and base-schedule -------
# Define the reconstruction function with mobility
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

        message["size"] += path_cost + mobility_factor

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

def generate_schedule_base_schedule(processing_times, message_list, all_path_indexes_with_costs):
    # Creating DEAP types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # DEAP toolbox
    toolbox = base.Toolbox()

    #parameters
    num_tasks = len(processing_times)
    nodes = json_data["platform"]["nodes"]
    processors = [node for node in nodes if not node["is_router"]]
    num_machines = len(processors) - 1
    num_message = len(message_list)
    num_paths = len(all_path_indexes_with_costs)

    # Define the initialization function for the individual
    def init_individual():
        return random.sample(range(num_tasks), num_tasks)

    # Register the initialization function in the DEAP toolbox
    toolbox.register("task_order", init_individual)

    def processor_allocation(n_task, n_machines):
        return [random.randint(0, num_machines) for _ in range(n_task)]

    toolbox.register("processor_allocation", processor_allocation, n_task=num_tasks, n_machines=num_machines)

    def message_path_index(n_messages, max_all_path_index):
        return [random.randint(0, max_all_path_index - 1) for _ in range(n_messages)]

    toolbox.register("message_path_index", message_path_index, n_messages=num_message, max_all_path_index=num_paths)

    def message_priority_ordering(n_messages):
        return random.sample(range(n_messages), n_messages)

    toolbox.register("message_priority_ordering", message_priority_ordering, n_messages=num_message)

    def create_individual():
        individual = []
        individual.extend(toolbox.task_order())
        individual.extend(toolbox.processor_allocation())
        individual.extend(toolbox.message_path_index())
        individual.extend(toolbox.message_priority_ordering())
        return individual

    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)

    # Population initialization
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Reconstruction function
    def reconstruct_schedule(processor_allocation, task_order):
        schedule = {i: [] for i in range(max(processor_allocation) + 1)}
        for task in task_order:
            schedule[processor_allocation[task]].append(task)
        return schedule

    # Evaluation function
    def evaluate(individual, processing_times, message_list, all_path_indexes_with_costs):
        task_order_len = num_tasks
        processor_allocation_len = num_tasks
        message_path_index_len = num_message
        message_priority_ordering_len = num_message
        task_order = individual[:task_order_len]
        processor_allocation = individual[task_order_len:task_order_len + processor_allocation_len]
        message_path_index = individual[task_order_len + processor_allocation_len : task_order_len + processor_allocation_len + message_path_index_len]
        message_priority_ordering = individual[task_order_len + processor_allocation_len + message_path_index_len :]
        schedule = reconstruct_schedule_with_mobility(processor_allocation, task_order, processing_times, message_list, message_path_index, all_path_indexes_with_costs, message_priority_ordering)
        makespan  = compute_makespan(schedule)
        fitness = 1.0 / makespan
        return fitness,

    toolbox.register("evaluate", evaluate, processing_times=processing_times,
                      message_list=message_list, all_path_indexes_with_costs=all_path_indexes_with_costs)

    # Your existing code...

    return scheduleFinal, Final_genome
