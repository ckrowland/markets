// zig fmt: off
pub const vs =
\\  @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) color: vec3<f32>,
\\  }
\\  @vertex fn main(
\\      @location(0) vertex_position: vec3<f32>,
\\      @location(1) position: vec4<f32>,
\\      @location(2) color: vec4<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      var x = position[0] + vertex_position[0];
\\      var y = position[1] + vertex_position[1];
\\      output.position_clip = vec4(x, y, 0.0, 1.0) * object_to_clip;
\\      output.color = color.xyz;
\\      return output;
\\  }
;
pub const producer_vs =
\\  @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) color: vec3<f32>,
\\  }
\\  @vertex fn main(
\\      @location(0) vertex_position: vec3<f32>,
\\      @location(1) position: vec4<f32>,
\\      @location(2) color: vec4<f32>,
\\      @location(3) inventory: u32,
\\      @location(4) max_inventory: u32,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      let num = f32(inventory) / f32(max_inventory);
\\      let scale = max(num, 0.4);
\\      var x = position[0] + (scale * vertex_position[0]);
\\      var y = position[1] + (scale * vertex_position[1]);
\\      output.position_clip = vec4(x, y, 0.0, 1.0) * object_to_clip;
\\      output.color = color.xyz;
\\      return output;
\\  }
;
pub const fs =
\\  @stage(fragment) fn main(
\\      @location(0) color: vec3<f32>,
\\  ) -> @location(0) vec4<f32> {
\\      return vec4(color, 1.0);
\\  }
;
pub const cs =
\\  struct Consumer {
\\    position: vec4<f32>,
\\    home: vec4<f32>,
\\    destination: vec4<f32>,
\\    step_size: vec4<f32>,
\\    color: vec4<f32>,
\\    moving_rate: f32,
\\    inventory: u32,
\\    radius: f32,
\\    producer_id: i32,
\\  }
\\  struct Producer {
\\    position: vec4<f32>,
\\    color: vec4<f32>,
\\    production_rate: u32,
\\    giving_rate: u32,
\\    inventory: atomic<u32>,
\\    max_inventory: u32,
\\    len: atomic<u32>,
\\    queue: array<u32, 450>,
\\  }
\\  struct Stats {
\\    transactions: u32,
\\  }
\\
\\  @group(0) @binding(0) var<storage, read_write> consumers: array<Consumer>;
\\  @group(0) @binding(1) var<storage, read_write> producers: array<Producer>;
\\  @group(0) @binding(2) var<storage, read_write> stats: Stats;
\\  @compute @workgroup_size(64)
\\  fn consumer_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
\\      let index : u32 = GlobalInvocationID.x;
\\      let nc = arrayLength(&consumers);
\\      if(GlobalInvocationID.x >= nc) {
\\        return;
\\      }
\\      let c = consumers[index];
\\      consumers[index].position += c.step_size;
\\      let dist = abs(c.position - c.destination);
\\      let at_destination = all(dist.xy <= vec2<f32>(0.1));
\\      
\\      if (at_destination) {
\\          var new_destination = vec4<f32>(0);
\\          let at_home = all(c.destination == c.home);
\\          if (at_home) {
\\              consumers[index].position = c.home;
\\              let consumption_rate = 1u;
\\              if (c.inventory >= consumption_rate) {
\\                  consumers[index].inventory -= consumption_rate;
\\                  consumers[index].destination = c.home;
\\                  consumers[index].step_size = vec4<f32>(0);
\\                  return;
\\              }
\\              consumers[index].color = vec4(1.0, 0.0, 0.0, 0.0);
\\              var closest_producer = vec4(10000.0, 10000.0, 0.0, 0.0);
\\              var shortest_distance = 100000.0;
\\              var array_len = i32(arrayLength(&producers));
\\              for(var i = 0; i < array_len; i++){
\\                  let dist = distance(c.home, producers[i].position);
\\                  let inventory = atomicLoad(&producers[i].inventory);
\\                  if (dist < shortest_distance && inventory > consumption_rate) {
\\                      shortest_distance = dist;
\\                      consumers[index].destination = producers[i].position;
\\                      consumers[index].step_size = step_sizes(c.position, producers[i].position, c.moving_rate);
\\                      consumers[index].producer_id = i;
\\                  }
\\              }
\\              if (shortest_distance == 100000.0) {
\\                  consumers[index].destination = c.home;
\\                  consumers[index].step_size = vec4<f32>(0);
\\              }
\\          } else {
\\              let position = c.destination;
\\              let pid = c.producer_id;
\\              consumers[index].position = position;
\\              consumers[index].step_size = vec4<f32>(0);
\\              let idx = atomicAdd(&producers[pid].len, 1);
\\              producers[pid].queue[idx] = index + 1;
\\          }
\\      }
\\  }
\\  fn step_sizes(pos: vec4<f32>, dest: vec4<f32>, mr: f32) -> vec4<f32>{
\\      let x_num_steps = num_steps(pos.x, dest.x, mr);
\\      let y_num_steps = num_steps(pos.y, dest.y, mr);
\\      let num_steps = max(x_num_steps, y_num_steps);
\\      let distance = dest - pos;
\\      return distance / num_steps;
\\  }
\\  fn num_steps(x: f32, y: f32, rate: f32) -> f32 {
\\      let distance = abs(x - y);
\\      if (rate > distance) { return 1.0; }
\\      return ceil(distance / rate);
\\  }
\\  @compute @workgroup_size(64)
\\  fn producer_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
\\      let index : u32 = GlobalInvocationID.x;
\\      let np = arrayLength(&producers);
\\      if(GlobalInvocationID.x >= np) {
\\        return;
\\      }
\\      let max_inventory = producers[index].max_inventory;
\\      let inventory = atomicLoad(&producers[index].inventory);
\\      var production_rate = producers[index].production_rate;
\\      let giving_rate = producers[index].giving_rate;
\\      if (max_inventory > inventory) {
\\          let diff = max_inventory - inventory;
\\          production_rate = min(diff, production_rate);
\\          let old_val = atomicAdd(&producers[index].inventory, production_rate);
\\      }
\\
\\      let idx = atomicLoad(&producers[index].len);
\\      for (var i = 0u; i < idx; i++) {
\\          let cid = producers[index].queue[i] - 1;
\\          let c = consumers[cid];
\\          let inventory = atomicLoad(&producers[index].inventory);
\\          if (inventory >= giving_rate) {
\\              consumers[cid].destination = c.home;
\\              consumers[cid].step_size = step_sizes(c.position, c.home, c.moving_rate);
\\              consumers[cid].inventory += giving_rate;
\\              let old_inv = atomicSub(&producers[index].inventory, giving_rate);
\\              stats.transactions += 1;
\\              consumers[cid].color = vec4(0.0, 1.0, 0.0, 0.0);
\\          }
\\      }
\\      atomicStore(&producers[index].len, 0);
\\}
;
// zig fmt: on
