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
\\    color: vec4<f32>,
\\    velocity: vec4<f32>,
\\    acceleration: vec4<f32>,
\\    consumption_rate: i32,
\\    jerk: f32,
\\    inventory: i32,
\\    radius: f32,
\\    producer_id: i32,
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
\\      let nc = arrayLength(&consumers);
\\      if(GlobalInvocationID.x >= nc) {
\\        return;
\\      }
\\      var velocity = c.velocity;
//\\      var velocity = c.velocity + c.acceleration;
\\      let new_pos = c.position + velocity;
\\
        //Wall[0] (or Wall.x) is the slope
        //Wall[1] (or Wall.y) is the y-intercept.
//\\      let wall = vec4<f32>(-1.0, -100.0, 0.0, 0.0);
//\\
//\\      let wall_pos_y = (wall.x * c.position.x) + wall.y;
//\\      let wall_pos_x = (wall_pos_y - wall.y) / wall.x;
//\\      let collision_x = abs(c.position.x - wall_pos_x) <= abs(velocity.x);
//\\      let collision_y = abs(c.position.y - wall_pos_y) <= abs(velocity.y);
//\\      if (collision_x && collision_y) {
//\\          let dist = c.radius;
//\\          velocity.x = -c.velocity.y;
//\\          velocity.y = 0;
//\\          consumers[index].position += velocity;
//\\      } else {
//\\          consumers[index].position += velocity;
//\\      }
\\      consumers[index].velocity = velocity;
\\  }
;
// zig fmt: on
