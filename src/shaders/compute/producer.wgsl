@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if (index >= stats.num_producers) {
        return;
    }
    let max_inventory = producers[index].max_inventory;
    let pc = producers[index].production_cost;
    var money = atomicLoad(&producers[index].money);

    let max_resources_can_produce = u32(money / pc);
    var inv = atomicLoad(&producers[index].inventory);
    let max_rate = min(max_resources_can_produce, max_inventory - inv);
    let rate = min(max_rate, producers[index].max_production_rate);

    atomicAdd(&producers[index].inventory, rate);
    atomicSub(&producers[index].money, rate * pc);
    money = atomicLoad(&producers[index].money);
    if (money > producers[index].max_money) {
        atomicStore(&producers[index].money, producers[index].max_money);
    }

    inv = atomicLoad(&producers[index].inventory);
    let dr = producers[index].decay_rate;
    if (inv >= dr) {
        atomicSub(&producers[index].inventory, dr);
    }
}
