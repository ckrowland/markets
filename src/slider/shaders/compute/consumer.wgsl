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
    let moving_rate = consumer_params.moving_rate;
    let demand_rate = consumer_params.demand_rate;

    consumers[index].position[0] += c.step_size[0];
    consumers[index].position[1] += c.step_size[1];

    let dist = abs(c.position - c.destination);
    let at_destination = all(dist.xy <= vec2<f32>(0.1));
    if (at_destination) {
        consumers[index].step_size = vec2<f32>(0);
        consumers[index].position = c.destination;
        let at_home = all(c.destination == c.home);
        if (at_home) {
            if (c.inventory >= u32(1)) {
                consumers[index].inventory -= u32(1);
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
        let old_val = atomicSub(&producers[pid].inventory, demand_rate);

        // Went negative, revert inventory
        if (demand_rate > old_val) {
            atomicAdd(&producers[pid].inventory, demand_rate);
            return;
        }

        consumers[index].color = vec4(0.0, 1.0, 0.0, 0.0);
        consumers[index].destination = c.home;
        consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
        consumers[index].inventory += u32(demand_rate);
        consumers[index].producer_id = -1;
        stats.transactions += u32(1);
    }
}

fn search_for_producer(index: u32){
    let c = consumers[index];

    var closest_producer = vec4(10000.0, 10000.0, 0.0, 0.0);
    var shortest_distance = 100000.0;
    var prev_shortest_distance = 0.0;
    var pid: i32 = -1;

    for(var i: u32 = 0; i < stats.num_producers; i++){
        shortest_distance = 100000.0;
        for (var j: u32 = 0; j < stats.num_producers; j++) {
            let dist = distance(c.home, producers[j].home);
            if (dist < shortest_distance && dist > prev_shortest_distance) {
                shortest_distance = dist;
                pid = i32(j);
            }
        }
        prev_shortest_distance = shortest_distance;

        let demand_rate = consumer_params.demand_rate;
        let pre_inventory = atomicSub(&producers[pid].available_inventory, demand_rate);

        // Not enough in inventory
        if (demand_rate > pre_inventory) {
            atomicAdd(&producers[pid].available_inventory, demand_rate);
        } else {
            consumers[index].destination = producers[pid].home;
            consumers[index].step_size = step_sizes(c.position.xy, producers[pid].home.xy, consumer_params.moving_rate);
            consumers[index].producer_id = pid;
            return;
        }
    }
}
