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
    let income = consumer_params.income;

    consumers[index].position[0] += c.step_size[0];
    consumers[index].position[1] += c.step_size[1];

    if (c.money + income <= c.max_money) {
        consumers[index].money += income;
    }

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
            if (consumer_params.demand_rate > 0) {
                search_for_producer(index);
            }
            return;
        }

        // At Producer
        let pid = c.producer_id;
        let demand_rate = min(consumer_params.demand_rate, i32(c.money / producers[pid].price));
        let cost = demand_rate * producers[pid].price;
        if (cost > c.money) {
            consumers[index].destination = c.home;
            consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
            consumers[index].producer_id = -1;
            return;
        }

        let pre_inventory = atomicSub(&producers[pid].inventory, demand_rate);
        if (demand_rate > pre_inventory) {
            atomicAdd(&producers[pid].inventory, demand_rate);
            consumers[index].destination = c.home;
            consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
            consumers[index].producer_id = -1;
            return;
        }

        consumers[index].color = vec4(0.0, 1.0, 0.0, 0.0);
        consumers[index].inventory += demand_rate;
        stats.transactions += u32(1);
        consumers[index].money -= cost;
        producers[pid].money += cost;

        consumers[index].destination = c.home;
        consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
        consumers[index].producer_id = -1;
    }
}

fn search_for_producer(index: u32){
    let c = consumers[index];
    var closest_producer = vec4(10000.0, 10000.0, 0.0, 0.0);
    var shortest_distance = 100000.0;
    var prev_shortest_distance = 0.0;
    var pid: i32 = -1;
    for (var j: u32 = 0; j < stats.num_producers; j++) {
        let dist = distance(c.home, producers[j].home);
        if (dist < shortest_distance && dist > prev_shortest_distance) {
            shortest_distance = dist;
            pid = i32(j);
        }
    }
    consumers[index].destination = producers[pid].home;
    consumers[index].step_size = step_sizes(c.position.xy, producers[pid].home.xy, consumer_params.moving_rate);
    consumers[index].producer_id = pid;
}
