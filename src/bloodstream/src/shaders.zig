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
pub const spline_vs =
\\  @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
\\  struct VertexOut {
\\      @builtin(position) position_clip: vec4<f32>,
\\      @location(0) color: vec3<f32>,
\\  }
\\  @vertex fn main(
\\      @location(0) vertex_position: vec3<f32>,
\\      @location(1) position: vec4<f32>,
\\      @location(2) radius: f32,
\\      @location(3) color: vec4<f32>,
\\  ) -> VertexOut {
\\      var output: VertexOut;
\\      var nx = vertex_position.x;
\\      var ny = vertex_position.y;
\\      let rad = 0.0;
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
\\  struct AnimatedSpline{
\\      current: array<SplinePoint, 10>,
\\      start: array<SplinePoint, 10>,
\\      end: array<SplinePoint, 10>,
\\      len: u32,
\\      to_start: u32,
\\  }
\\  struct Spline{
\\      points: array<SplinePoint, 1000>,
\\  } 
\\  struct SplinePoint{
\\      color: vec4<f32>,
\\      position: vec4<f32>,
\\      radius: f32,
\\      step_size: f32,
\\  }
\\  const PI: f32 = 3.14159;
\\  @group(0) @binding(0) var<storage, read_write> consumers: array<Consumer>;
\\  @group(0) @binding(1) var<storage, read_write> stats: vec4<i32>;
\\  @group(0) @binding(2) var<storage, read> size: Size;
\\  @group(0) @binding(3) var<storage, read_write> animated_splines: array<AnimatedSpline>;
\\  @group(0) @binding(4) var<storage, read_write> splines: array<Spline>;
\\  @compute @workgroup_size(64)
\\  fn consumer_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
\\      let index : u32 = global_id.x;
\\      var src_ball = consumers[index];
\\      let num_consumers = arrayLength(&consumers);
\\      if(index >= num_consumers) {
\\        return;
\\      }
\\      let num_splines = arrayLength(&animated_splines);
\\      for (var i = 0u; i < num_splines; i += 1u) {
\\          let s = animated_splines[i];
\\          let num_curves = s.len - 3;
\\          let diameter = s.current[i].radius * 0.001;
\\          let overlap = diameter * 5;
\\          for (var i = 0u; i < num_curves; i += 1u) {
\\              let points = getCurvePoints(s.current, i);
\\              for (var i = 0.0; i <= 1; i += diameter) {
\\                  src_ball = updateIfCircleCollision(src_ball,
\\                                                     i,
\\                                                     i + diameter + overlap,
\\                                                     points);
\\              }
\\          }
\\      }
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
\\
\\  fn getCurvePoints(spline: array<SplinePoint, 10>, i: u32) -> mat2x4<f32> {
\\      let p0 = spline[i].position;
\\      let p1 = spline[i + 1].position;
\\      let p2 = spline[i + 2].position;
\\      let p3 = spline[i + 3].position;
\\      return mat2x4(p0.x, p1.x, p2.x, p3.x, p0.y, p1.y, p2.y, p3.y);
\\  }
\\  fn updateIfCircleCollision(ball: Consumer, start_t: f32, end_t: f32, points: mat2x4<f32>) -> Consumer {
\\      var updated_ball = ball;
\\      let n_end_t = min(1, end_t);
\\      let start_edge = calculateSplinePoint(start_t, points).xy;
\\      let end_edge = calculateSplinePoint(n_end_t, points).xy;
\\      let radius = distance(start_edge, end_edge) / 2;
\\      let to_center = (end_edge - start_edge) / 2;
\\      let center_pos = start_edge + to_center;
\\      let n = ball.position.xy - center_pos;
\\      let distance = length(n);
\\      if (distance < radius) {
\\          let overlap = ball.radius + radius - distance;
\\          let new_pos = ball.position.xy + normalize(n) * overlap;
\\          updated_ball.position = vec4(new_pos.x, new_pos.y, 0.0, 0.0);
\\
\\          let ball_mass = pow(ball.radius, 2.0) * PI;
\\          let other_mass = ball_mass;
\\          let other_velocity = -ball.velocity.xy;
\\          let c = 2.*dot(n, (other_velocity - ball.velocity.xy)) / (dot(n, n) * (1./ball_mass + 1./other_mass));
\\          let new_vel = ball.velocity.xy + c/ball_mass * n;
\\          updated_ball.velocity = vec4(new_vel.x, new_vel.y, 0.0, 0.0);
\\      }
\\      return updated_ball;
\\  }
\\  fn calculateSplinePoint(t: f32, points: mat2x4<f32>) -> vec4<f32> {
\\      let tt = t * t;
\\      let ttt = tt * t;
\\      let q1 = -ttt + (2 * tt) - t;
\\      let q2 = (3 * ttt) - (5 * tt) + 2;
\\      let q3 = (-3 * ttt) + (4 * tt) + t;
\\      let q4 = ttt - tt;
\\      let influence = vec4(q1, q2, q3, q4);
\\      let result_x = dot(points[0], influence);
\\      let result_y = dot(points[1], influence);
\\      let sx = 0.5 * result_x;
\\      let sy = 0.5 * result_y;
\\      return vec4(sx, sy, 0, 0);
\\  }
\\
\\  @compute @workgroup_size(64)
\\  fn animated_spline_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
\\      let index: u32 = global_id.x; 
\\      let num_splines = arrayLength(&animated_splines);
\\      if (index > num_splines) {
\\          return;
\\      }
\\
\\      let aspline = animated_splines[index];
\\      for (var i = 1u; i < aspline.len - 1; i += 1u) {
\\          let start = aspline.start[i].position;
\\          let current = aspline.current[i].position;
\\          let end = aspline.end[i].position;
\\          if (all(current.xy == start.xy) && all(current.xy == end.xy)) {
\\              continue;
\\          }
\\          var diff = end - current;
\\          if (all(current.xy == end.xy)) {
\\              animated_splines[index].to_start = 1;
\\          }
\\          if (all(current.xy == start.xy)) {
\\              animated_splines[index].to_start = 0;
\\          }
\\          if (animated_splines[index].to_start == 1) {
\\              diff = start - current;
\\          }
\\          let direction = normalize(diff) * aspline.current[i].step_size;
\\          animated_splines[index].current[i].position += direction;
\\
\\          let points_idx = i - 1;
\\          let points = getCurvePoints(animated_splines[index].current, points_idx);
\\          for (var j = 0u; j < 1000; j += 1) {
\\              let t = f32(j) * 0.001;
\\              let new_point = calculateSplinePoint(t, points);
\\              splines[points_idx].points[j].position = new_point;
\\          }
\\      }
\\  }
;
// zig fmt: on
