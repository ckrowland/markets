@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if (index >= stats.num_producers) {
        return;
    }
    let max_inventory = producers[index].max_inventory;
    let pc = producers[index].production_cost;
    var money = atomicLoad(&producers[index].money);
    var max_money = producers[index].max_money;

    let max_resources_can_produce = u32(money / pc);
    var inv = atomicLoad(&producers[index].inventory);
    let max_rate = min(max_resources_can_produce, max_inventory - inv);
    let rate = min(max_rate, producers[index].max_production_rate);
    atomicAdd(&producers[index].inventory, rate);
    atomicSub(&producers[index].money, rate * pc);

    money = atomicLoad(&producers[index].money);
    if (money > max_money) {
        atomicStore(&producers[index].money, max_money);
    }

    inv = atomicLoad(&producers[index].inventory);
    let dr = producers[index].decay_rate;
    if (inv >= dr) {
        atomicSub(&producers[index].inventory, dr);
    }

    if (producers[index].price > 10) {
        producers[index].price -= 10;
    }

    let inv_close = abs(max_inventory - inv) < 100;
    let money_close = abs(max_money - money) < 100;
    if (inv_close && money_close) {
        let p = &producers[index];

        let pos = get_clear_home((*p).home);
        if (!pos.valid) {
            return;
        }

        var new_home = pos.pos;
        let new_p_ptr = &producers[stats.num_producers];
        (*new_p_ptr).home = new_home;
        (*new_p_ptr).color = (*p).color;
        atomicStore(&(*new_p_ptr).inventory,  (*p).max_inventory / 2);
        (*new_p_ptr).max_inventory = (*p).max_inventory;
        atomicStore(&(*new_p_ptr).money,  money / 2);
        (*new_p_ptr).max_money = (*p).max_money;
        (*new_p_ptr).price = (*p).price;
        (*new_p_ptr).production_cost = (*p).production_cost;
        (*new_p_ptr).max_production_rate = (*p).max_production_rate;
        (*new_p_ptr).decay_rate = (*p).decay_rate;

        atomicStore(&producers[index].inventory,  inv / 2);
        atomicStore(&producers[index].money, money / 2);
        stats.num_producers += 1;
    }
}

fn set_white(i: u32) {
    producers[i].color = vec4(1, 1, 1, 0);
}

fn set_blue_if(i: u32, cmp: bool) {
    if (cmp) {
        producers[i].color = vec4(0, 0, 1, 0);
    }
}

struct getPos {
    valid: bool,
    pos: vec4<f32>,
}
fn get_clear_home(pos: vec4<f32>) -> getPos {
    var rand_x = stats.rand_num.x * f32(stats.max_x);
    var rand_y = stats.rand_num.y * f32(stats.max_y);
    var new_home = pos;
    new_home.x = rand_x;
    new_home.y = rand_y;
    for (var i: u32 = 0; i < stats.num_producers; i++) {
        if (distance(new_home, producers[i].home) < 10) {
            return getPos(false, new_home);
        }
    }
    return getPos(true, new_home);
}
