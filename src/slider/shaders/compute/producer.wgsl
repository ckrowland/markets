@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if (index >= stats.num_producers) {
      return;
    }
    let max_inventory = producers[index].max_inventory;
    let max_production = i32(producers[index].money / producers[index].production_cost);
    let inv = atomicLoad(&producers[index].inventory);
    let rate = min(max_production, max_inventory - inv);

    let old_inv = atomicAdd(&producers[index].inventory, rate);
    if (old_inv + rate > max_inventory) {
        atomicSub(&producers[index].inventory, rate);
        return;
    }
    producers[index].money -= rate * producers[index].production_cost;
}
