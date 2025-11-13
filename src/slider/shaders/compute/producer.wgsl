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
    let inv = atomicLoad(&producers[index].inventory);
    if (inv > max_inventory) {
        atomicStore(&producers[index].inventory, max_inventory);
        return;
    }
    let rate = min(max_resources_can_produce, max_inventory - inv);
    atomicAdd(&producers[index].inventory, rate);

    atomicSub(&producers[index].money, rate * pc);
    money = atomicLoad(&producers[index].money);
    if (money > producers[index].max_money) {
        atomicStore(&producers[index].money, producers[index].max_money);
    }

    // Can we sub beyond 0 on a u32?
    atomicSub(&producers[index].inventory, producers[index].decay_rate);
}
