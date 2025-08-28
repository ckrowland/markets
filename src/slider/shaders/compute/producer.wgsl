@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if(index >= stats.num_producers) {
      return;
    }
    let max_inventory = producers[index].max_inventory;
    var production_rate = producers[index].production_rate;

    let old_a_inventory = atomicAdd(&producers[index].available_inventory, production_rate);
    let old_inventory = atomicAdd(&producers[index].inventory, production_rate);

    if (old_a_inventory + production_rate > max_inventory) {
        atomicStore(&producers[index].available_inventory, max_inventory);
    }

    if (old_inventory + production_rate > max_inventory) {
        atomicStore(&producers[index].inventory, max_inventory);
    }
}
