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
\\      @location(3) inventory: i32,
\\      @location(4) max_inventory: i32,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      let num = f32(inventory) / f32(max_inventory);
\\      let scale = num + 0.5;
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
\\    center: vec4<f32>,
\\    color: vec4<f32>,
\\    consumption_rate: i32,
\\    moving_rate: f32,
\\    inventory: i32,
\\    radius: f32,
\\    producer_id: i32,
\\    angle: f32,
\\    center_radius: f32,
\\  }
\\  struct Stats {
\\    second: i32,
\\    num_transactions: i32,
\\    num_empty_consumers: i32,
\\    num_total_producer_inventory: i32,
\\  }
\\
\\  @group(0) @binding(0) var<storage, read_write> consumers: array<Consumer>;
\\  @group(0) @binding(1) var<storage, read_write> stats: vec4<i32>;
\\  @compute @workgroup_size(64)
\\  fn consumer_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
\\      let index : u32 = GlobalInvocationID.x;
\\      let c = consumers[index];
\\      let rad = radians(c.angle);
\\      var new_pos = vec4(cos(rad), sin(rad), 0.0, 0.0);
\\      new_pos *= c.center_radius;
\\      new_pos += c.center;
\\      consumers[index].position = new_pos;
\\      consumers[index].angle += c.moving_rate;
\\      consumers[index].moving_rate = max(c.moving_rate - 0.5, 0);
\\
\\      if (stats[0] % 3 == 1) {
\\          consumers[index].moving_rate = 28.0;
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
;
// zig fmt: on
