@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if (index >= stats.num_consumers) {
        return;
    }

    // User removed producer this consumer was targeting
    if (consumers[index].producer_id >= i32(stats.num_producers)) {
        search_for_producer(index);
    }

    let c = consumers[index];
    let moving_rate = consumer_params[c.grouping_id].moving_rate;
    let demand_rate = consumer_params[c.grouping_id].demand_rate;
    consumers[index].position[0] += c.step_size[0];
    consumers[index].position[1] += c.step_size[1];

    let dist = abs(c.position - c.destination);
    let at_destination = all(dist.xy <= vec2<f32>(0.1));
    if (at_destination) {
        consumers[index].step_size = vec2<f32>(0);
        consumers[index].position = c.destination;
        let at_home = all(c.destination == c.home);
        if (at_home) {
            if (c.inventory >= 1) {
                consumers[index].inventory -= 1;
                return;
            }
            consumers[index].color = vec4(1.0, 0.0, 0.0, 0.0);
            if (demand_rate > 0) {
                search_for_producer(index);
            }
            return;
        }

        // At Producer
        let pid = c.producer_id;
        let i_demand_rate = i32(demand_rate);
        let old_val = atomicSub(&producers[pid].inventory, i_demand_rate);

        // Went negative, revert inventory
        if (i_demand_rate > old_val) {
            atomicAdd(&producers[pid].inventory, i_demand_rate);
            return;
        }
        consumers[index].color = vec4(0.0, 1.0, 0.0, 0.0);
        consumers[index].destination = c.home;
        consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
        consumers[index].inventory += demand_rate;
        consumers[index].producer_id = -1;
        stats.transactions += 1;
    }
}

fn search_for_producer(index: u32){
    let c = consumers[index];
    let moving_rate = consumer_params[c.grouping_id].moving_rate;
    var pid = find_nearest_stocked_producer(c);
    if (pid == -1) {
        consumers[index].destination = c.home;
        consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
        return;
    }
    let p_pos = producers[pid].home;  
    consumers[index].destination = p_pos;
    consumers[index].step_size = step_sizes(c.position.xy, p_pos.xy, moving_rate);
    consumers[index].producer_id = pid;
}

// Returns the pid of nearest stocked producer, -1 for failure
fn find_nearest_stocked_producer(c: Consumer) -> i32 {
    var closest_producer = vec4(10000.0, 10000.0, 0.0, 0.0);
    var shortest_distance = 100000.0;
    var pid: i32 = -1;
    let demand_rate = consumer_params[c.grouping_id].demand_rate;
    for(var i: u32 = 0; i < stats.num_producers; i++){
        let dist = distance(c.home, producers[i].home);
        let inventory = u32(atomicLoad(&producers[i].inventory));
        if (dist < shortest_distance && inventory > demand_rate) {
            shortest_distance = dist;
            pid = i32(i);
        }
    }
    return pid;
}
