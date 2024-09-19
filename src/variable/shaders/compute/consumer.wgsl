@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if(GlobalInvocationID.x >= stats.num_consumers) {
        return;
    }

    // User removed producer this consumer was targeting
    if (consumers[index].producer_id >= i32(stats.num_producers)) {
        search_for_producer(index);
    }

    let c = consumers[index];
    let moving_rate = consumer_params.moving_rate;
    let max_demand_rate = consumer_params.max_demand_rate;
    let income = consumer_params.income;

    consumers[index].position[0] += c.step_size[0];
    consumers[index].position[1] += c.step_size[1];
    if (c.balance + income < c.max_balance) {
        consumers[index].balance += income;
    }

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

            let demand_rate = min(c.balance, max_demand_rate);
            if (demand_rate > 0) {
              search_for_producer(index);
            }
            return;
        }


        // At Producer
        let pid = c.producer_id;
        let max_consumer_can_buy = c.balance / producers[pid].params.price;
        let demand_rate = i32(min(max_consumer_can_buy, max_demand_rate));
        let purchase_price = u32(demand_rate) * producers[pid].params.price;

        let old_inventory = atomicSub(&producers[pid].inventory, demand_rate);
        if (old_inventory < demand_rate) {
          _ = atomicAdd(&producers[pid].inventory, demand_rate);
          return;
        }

        _ = atomicAdd(&producers[pid].balance, i32(purchase_price));
        producers[pid].params.num_sales += u32(1);

        consumers[index].color = vec4(0.0, 1.0, 0.0, 0.0);
        consumers[index].destination = c.home;
        consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
        consumers[index].inventory += u32(demand_rate);
        consumers[index].balance -= purchase_price;
        consumers[index].producer_id = -1;
        stats.transactions += u32(1);
    }
}

fn search_for_producer(index: u32){
    let c = consumers[index];
    let moving_rate = consumer_params.moving_rate;
    var pid = find_nearest_stocked_producer(c, index);
    if (pid == -1) {
        consumers[index].destination = c.home;
        consumers[index].step_size = step_sizes(
            c.position.xy, 
            c.home.xy, 
            moving_rate,
        );
        return;
    }
    let p_pos = producers[pid].params.home;  
    consumers[index].destination = p_pos;
    consumers[index].step_size = step_sizes(c.position.xy, p_pos.xy, moving_rate);
    consumers[index].producer_id = pid;
}

// Returns the pid of nearest stocked producer, -1 for failure
fn find_nearest_stocked_producer(c: Consumer, index: u32) -> i32 {
    var closest_producer = vec4(10000.0, 10000.0, 0.0, 0.0);
    var shortest_distance = 100000.0;
    var pid: i32 = -1;
    for(var i: u32 = 0; i < stats.num_producers; i++){
        let dist = distance(c.home, producers[i].params.home);
        let inventory = atomicLoad(&producers[i].inventory);
        let max_consumer_can_buy = c.balance / producers[pid].params.price;
        let demand_rate = i32(min(max_consumer_can_buy, consumer_params.max_demand_rate));
        if (dist < shortest_distance && inventory > demand_rate) {
            shortest_distance = dist;
            pid = i32(i);
        }
    }
    return pid;
}

fn step_sizes(pos: vec2<f32>, dest: vec2<f32>, mr: f32) -> vec2<f32>{
    let x_num_steps = num_steps(pos.x, dest.x, mr);
    let y_num_steps = num_steps(pos.y, dest.y, mr);
    let num_steps = max(x_num_steps, y_num_steps);
    let distance = dest - pos;
    return distance / num_steps;
}
fn num_steps(x: f32, y: f32, rate: f32) -> f32 {
    let distance = abs(x - y);
    if (rate > distance) { return 1.0; }
    return ceil(distance / rate);
}
