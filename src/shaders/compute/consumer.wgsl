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
    let moving_rate = c.moving_rate;
    let income = c.income;

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
            consumers[index].color = red;
            search_for_producer(index);
            return;
        }

        // At Producer
        let pid = c.producer_id;
        var inv = atomicLoad(&producers[pid].inventory);
        var total_can_buy = c.money / producers[pid].price;
        var can_buy = min(inv, total_can_buy);
        if (can_buy < 1) {
            //go_home(index);
            return;
        }

        var result = inv - can_buy;
        var cmp = atomicCompareExchangeWeak(&producers[pid].inventory, inv, result);
        while (!cmp.exchanged) {
            inv = atomicLoad(&producers[pid].inventory);
            can_buy = min(inv, total_can_buy);
            result = inv - can_buy;
            cmp = atomicCompareExchangeWeak(&producers[pid].inventory, inv, result);
        }
        if (result < 0) {
            producers[pid].color = vec4(1, 0, 0, 1);
        }
        if (can_buy < 0) {
            producers[pid].color = vec4(1, 0, 0, 1);
        }
        buy(index, producers[pid].price, can_buy);
        go_home(index);
        return;
    }
}

fn buy(index: u32, price: u32, amount: u32) {
    let cost = price * amount;
    let pid = consumers[index].producer_id;
    consumers[index].money -= cost;
    consumers[index].inventory += amount;
    consumers[index].color = green;
    stats.transactions += u32(1);
    atomicAdd(&producers[pid].money, cost);
}

fn go_home(index: u32) {
    let c = consumers[index];
    consumers[index].destination = c.home;
    consumers[index].step_size = step_sizes(c.position.xy, c.home.xy, c.moving_rate);
}

struct Metrics {
    amount_can_buy: u32,
    distance: f32,
}
fn search_for_producer(index: u32) {
    let c = consumers[index];
    var metrics: array<Metrics, MAX_NUM_PRODUCERS>;
    for (var i: u32 = 0; i < stats.num_producers; i++) {
        var inv = atomicLoad(&producers[i].inventory);
        metrics[i].amount_can_buy = inv / producers[i].price;
        metrics[i].distance = distance(c.home, producers[i].home);
    }

    var largest_transaction: u32 = 0;
    var pid: u32 = 0;
    for (var i: u32 = 0; i < stats.num_producers; i++) {
        if (metrics[i].amount_can_buy == largest_transaction &&
            metrics[i].distance < metrics[pid].distance) {
            pid = i;
        }
        if (metrics[i].amount_can_buy > largest_transaction) {
            largest_transaction = metrics[i].amount_can_buy;
            pid = i;
        }
    }

    if (largest_transaction == 0) { return; }

    consumers[index].destination = producers[pid].home;
    consumers[index].step_size = step_sizes(c.position.xy, producers[pid].home.xy, c.moving_rate);
    consumers[index].producer_id = pid;
}

fn find_shortest_producer(index: u32) -> u32 {
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
    return pid;
}
