
def create_task_data(task_id, processing_times, mcet, deadline, can_run_on):
    task_data = {
        "id": task_id,
        "processing_times": processing_times,
        "mcet": mcet,
        "deadline": deadline,
        "can_run_on": can_run_on
    }
    return task_data

# Example usage:
task_data_example = create_task_data(10, 2, 10, 256, [0, 1, 2])
print(task_data_example)


