@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let index : u32 = GlobalInvocationID.x;
    if(index >= stats.num_producers) {
      return;
    }

    let p = producers[index].params;
    producers[index].params.price = p.cost + p.margin;

    let production_rate = 100;
    let old_inventory = atomicAdd(&producers[index].inventory, production_rate);
    let too_high_inv = old_inventory + production_rate > i32(p.max_inventory);

    let price = i32(producers[index].params.price) * production_rate;
    let old_balance = atomicSub(&producers[index].balance, price);
    let too_low_bal = old_balance < price;

    if (too_high_inv || too_low_bal) {
        _ = atomicSub(&producers[index].inventory, production_rate);
        _ = atomicAdd(&producers[index].balance, price);
    }
}
