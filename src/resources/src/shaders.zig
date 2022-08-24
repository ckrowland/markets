// zig fmt: off
pub const producer_vs =
\\  @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) color: vec3<f32>,
\\  }
\\  @stage(vertex) fn main(
\\      @location(0) vertex_position: vec3<f32>,
\\      @location(1) color: vec3<f32>,
\\      @location(2) position: vec2<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      var x = position[0] + vertex_position[0];
\\      var y = position[1] + vertex_position[1];
\\      output.position_clip = vec4(x, y, 0.0, 1.0) * object_to_clip;
\\      output.color = color;
\\      return output;
\\  }
;
pub const consumer_vs =
\\  @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) color: vec3<f32>,
\\  }
\\  @stage(vertex) fn main(
\\      @location(0) vertex_position: vec3<f32>,
\\      @location(1) color: vec3<f32>,
\\      @location(2) position: vec2<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      var x = position[0] + vertex_position[0];
\\      var y = position[1] + vertex_position[1];
\\      output.position_clip = vec4(x, y, 0.0, 1.0) * object_to_clip;
\\      output.color = color;
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
\\  struct Position {
\\    position: vec2<f32>,
\\  }
\\  struct Consumer {
\\    position: vec4<f32>,
\\    home: vec4<f32>,
\\    destination: vec4<f32>,
\\    step_size: vec4<f32>,
\\    consumption_rate: i32,
\\    moving_rate: f32,
\\    inventory: i32,
\\    radius: f32,
\\  }
\\  struct Producer {
\\    position: vec4<f32>,
\\    production_rate: i32,
\\    giving_rate: i32,
\\    inventory: u32,
\\    width: f32,
\\  }
\\  @group(0) @binding(0) var<storage, read_write> consumers_a: array<Consumer>;
\\  @group(0) @binding(1) var<storage, read> producers: array<Producer>;
\\  @group(0) @binding(2) var<storage, read_write> transactions_this_second: f32;
\\  @compute @workgroup_size(64)
\\  fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
\\      let index : u32 = GlobalInvocationID.x;
\\      let c = consumers_a[index];
\\      consumers_a[index].position = c.position + c.step_size;
\\      let dist = abs(c.position - c.destination);
\\      let at_destination = all(dist.xy <= vec2<f32>(c.moving_rate / 2));
\\      
\\      if (at_destination) {
\\          var new_destination = vec4<f32>(0);
\\          let at_home = all(c.destination == c.home);
\\          if (at_home) {
\\              var closest_producer = vec4(10000.0, 10000.0, 0.0, 0.0);
\\              var shortest_distance = 100000.0;
\\              var array_len = i32(arrayLength(&producers));
\\              for(var i = 0; i < array_len; i++){
\\                  let dist = distance(c.home, producers[i].position);
\\                  if (dist < shortest_distance) {
\\                      shortest_distance = dist;
\\                      closest_producer = producers[i].position;
\\                  }
\\              }
\\              new_destination = closest_producer;
\\          } else {
\\              new_destination = c.home;
\\          }
\\          consumers_a[index].destination = new_destination;
\\          consumers_a[index].step_size = step_sizes(c.position, new_destination, c.moving_rate);
\\          transactions_this_second += 1;
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
\\      return distance / rate;
\\  }
;
// zig fmt: on
