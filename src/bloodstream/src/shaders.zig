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
\\      var x = position.x + vertex_position.x;
\\      var y = position.y + vertex_position.y;
\\      output.position_clip = vec4(x, y, 0.0, 1.0) * object_to_clip;
\\      output.color = color.xyz;
\\      return output;
\\  }
;
pub const line_vs =
\\  @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) color: vec3<f32>,
\\  }
\\  @vertex fn main(
\\      @location(0) vertex_position: vec3<f32>,
\\      @location(1) position: vec4<f32>,
\\      @location(2) color: vec4<f32>,
\\      @location(3) radius: f32,
\\      @location(4) rad: f32,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      var nx = vertex_position.x;
\\      var ny = vertex_position.y;
\\      if (vertex_position.x > 0) {
\\          nx += radius;
\\      } else {
\\          nx -= radius; 
\\      }
\\      if (vertex_position.y > 0) {
\\          ny += radius;
\\      } else {
\\          ny -= radius; 
\\      }
\\      var x = position.x + cos(rad) * nx - sin(rad) * ny;
\\      var y = position.y + sin(rad) * nx + cos(rad) * ny;
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
\\    consumption_rate: i32,
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
\\  struct Size{
\\    min_x: f32,
\\    min_y: f32,
\\    max_x: f32,
\\    max_y: f32,
\\  }
\\  struct Line{
\\    color: vec4<f32>,
\\    start: vec2<f32>,
\\    end: vec2<f32>,
\\    radius: f32,
\\    num_squares: u32,
\\  }
\\  const PI: f32 = 3.14159;
\\  @group(0) @binding(0) var<storage, read_write> consumers: array<Consumer>;
\\  @group(0) @binding(1) var<storage, read_write> stats: vec4<i32>;
\\  @group(0) @binding(2) var<storage, read> size: Size;
\\  @group(0) @binding(3) var<storage, read_write> lines: array<Line>;
\\  @compute @workgroup_size(64)
\\  fn consumer_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
\\      let index : u32 = global_id.x;
\\      var src_ball = consumers[index];
\\      let num_consumers = arrayLength(&consumers);
\\      if(index >= num_consumers) {
\\        return;
\\      }
\\      let num_lines = arrayLength(&lines);
\\      for (var i = 0u; i < num_lines; i += 1u) {
\\          let l = lines[i];
\\          let d_line = l.end - l.start;
\\          let line_len = d_line.x * d_line.x + d_line.y * d_line.y;
\\          let pos_start = src_ball.position.xy - l.start;
\\          let t = max(0, min(line_len, dot(d_line, pos_start))) / line_len;
\\          let closest_point = l.start + (t * d_line);
\\          let n = src_ball.position.xy - closest_point;
\\          let distance = length(n);
\\          if (distance >= src_ball.radius + l.radius) {
\\              continue;
\\          }
\\          //COLLISION
\\          let overlap = src_ball.radius + l.radius - distance;
\\          let new_pos = src_ball.position.xy + normalize(n) * overlap;
\\          src_ball.position = vec4(new_pos.x, new_pos.y, 0.0, 0.0);
\\
\\          let src_mass = pow(src_ball.radius, 2.0) * PI;
\\          let other_mass = src_mass;
\\          let other_velocity = -src_ball.velocity;
\\          let c = 2.*dot(n, (other_velocity.xy - src_ball.velocity.xy)) / (dot(n, n) * (1./src_mass + 1./other_mass));
\\          let new_vel = src_ball.velocity.xy + c/src_mass * n;
\\          src_ball.velocity = vec4(new_vel.x, new_vel.y, 0.0, 0.0);
\\      }
\\
\\      for (var i = 0u; i < num_consumers; i += 1u) {
\\          if (i == index) {
\\              continue;
\\          }
\\          var other_ball = consumers[i];
\\          let n = src_ball.position.xy - other_ball.position.xy;
\\          let distance = length(n);
\\          if (distance >= src_ball.radius + other_ball.radius) {
\\              continue;
\\          }
\\          //COLLISION
\\          let overlap = src_ball.radius + other_ball.radius - distance;
\\          let new_pos = src_ball.position.xy + normalize(n) * overlap/2.;
\\          src_ball.position = vec4(new_pos.x, new_pos.y, 0.0, 0.0);
\\
\\          let src_mass = pow(src_ball.radius, 2.0) * PI;
\\          let other_mass = pow(other_ball.radius, 2.0) * PI;
\\          let c = 2.*dot(n, (other_ball.velocity.xy - src_ball.velocity.xy)) / (dot(n, n) * (1./src_mass + 1./other_mass));
\\          let new_vel = src_ball.velocity.xy + c/src_mass * n;
\\          src_ball.velocity = vec4(new_vel.x, new_vel.y, 0.0, 0.0);
\\      }
\\      var velocity = src_ball.velocity;
\\      var position = src_ball.position + velocity;
\\      if(position.x - src_ball.radius <= size.min_x){
\\          position.x = size.min_x + src_ball.radius;
\\          velocity.x = -src_ball.velocity.x;
\\      }
\\      if(position.y - src_ball.radius <= size.min_y){
\\          position.y = size.min_y + src_ball.radius;
\\          velocity.y = -src_ball.velocity.y;
\\      }
\\      if(position.x + src_ball.radius >= size.max_x){
\\          position.x = size.max_x - src_ball.radius;
\\          velocity.x = -src_ball.velocity.x;
\\      }
\\      if(position.y + src_ball.radius >= size.max_y){
\\          position.y = size.max_y - src_ball.radius;
\\          velocity.y = -src_ball.velocity.y;
\\      }
\\      consumers[index].position = position;
\\      consumers[index].velocity = velocity;
\\  }
;
// zig fmt: on
