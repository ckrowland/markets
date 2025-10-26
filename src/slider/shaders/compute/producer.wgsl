@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if (index >= stats.num_producers) {
      return;
    }
    let max_inventory = producers[index].max_inventory;
    let pc = producers[index].production_cost;

    if (pc == 0) {
        atomicStore(&producers[index].inventory, max_inventory);
        return;
    }

    let max_resources_can_produce = u32(producers[index].money / pc);
    let inv = atomicLoad(&producers[index].inventory);
    let rate = min(max_resources_can_produce, max_inventory - inv);
    atomicAdd(&producers[index].inventory, rate);
    producers[index].money -= rate * producers[index].production_cost;
}
