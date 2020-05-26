def pruning_schedule(initial_value, final_value, num_steps, schedule_type="linear"):
    if schedule_type == "linear":
        schedule = [0 for _ in range(num_steps)]
        i = 0
        while sum(schedule) < initial_value - final_value:
            schedule[i%len(schedule)] += 1
            i += 1
    if schedule_type == "geometric":
        alpha = (float(final_value)/initial_value)**(1/(num_steps))
        schedule = []
        for t in range(num_steps):
            schedule.append( round((1-alpha) * alpha**t * initial_value) )
        i = 0
        while sum(schedule) < initial_value - final_value: # The sum can be different because of rounding
            schedule[i%len(schedule)] += 1
            i += 1
        i = len(schedule)-1
        while sum(schedule) > initial_value - final_value: # The sum can be different because of rounding
            schedule[i%len(schedule)] -= 1
            i -= 1
    return schedule
