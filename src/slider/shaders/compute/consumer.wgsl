@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if (index >= stats.num_consumers) {
        return;
    }

    // User removed producer this consumer was targeting
    if (consumers[index].producer_id >= stats.num_producers) {
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
            search_for_producer(index);
            return;
        }

        // At Producer
        let pid = c.producer_id;
        var total_can_buy = c.money / producers[pid].price;
        if (total_can_buy < 1) {
            go_home(index);
            return;
        }

        var inv = atomicLoad(&producers[pid].inventory);
        if (total_can_buy <= inv) {
            var result = inv - total_can_buy;
            var cmp = atomicCompareExchangeWeak(&producers[pid].inventory, inv, result);
            while (!cmp.exchanged) {
                if (cmp.old_value < total_can_buy) {
                    break;
                }
                inv = atomicLoad(&producers[pid].inventory);
                result = inv - total_can_buy;
                cmp = atomicCompareExchangeWeak(&producers[pid].inventory, inv, result);
            }
            if (cmp.exchanged) {
                var cost = total_can_buy * producers[pid].price;
                buy(index, cost, total_can_buy);
                go_home(index);
                return;
            }
        }

        inv = atomicLoad(&producers[pid].inventory);
        var cmp = atomicCompareExchangeWeak(&producers[pid].inventory, inv, 0);
        while (!cmp.exchanged) {
            inv = atomicLoad(&producers[pid].inventory);
            cmp = atomicCompareExchangeWeak(&producers[pid].inventory, inv, 0);
        }
        let cost = inv * producers[pid].price;
        buy(index, cost, inv);
        go_home(index);
    }
}

fn buy(index: u32, cost: u32, amount: u32) {
    let pid = consumers[index].producer_id;
    consumers[index].money -= cost;
    producers[pid].money += cost;
    consumers[index].inventory += amount;
    consumers[index].color = vec4(0.0, 1.0, 0.0, 0.0);
    stats.transactions += u32(1);
}

fn go_home(index: u32) {
    let c = consumers[index];
    let moving_rate = consumer_params.moving_rate;
    consumers[index].destination = c.home;
    consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, moving_rate);
}

fn search_for_producer(index: u32) {
    let c = consumers[index];
    var shortest_producer: f32 = 1000000;
    var pid: u32 = 0;
    for (var i: u32 = 0; i < stats.num_producers; i++) {
        let dist = distance(c.home, producers[i].home);
        if (dist < shortest_producer) {
            shortest_producer = dist;
            pid = i;
        }
    }

    consumers[index].destination = producers[pid].home;
    consumers[index].step_size = step_sizes(c.position.xy, producers[pid].home.xy, consumer_params.moving_rate);
    consumers[index].producer_id = pid;
}
