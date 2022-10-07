struct Consumer {
  position: vec4<f32>,
  color: vec4<f32>,
  velocity: vec4<f32>,
  consumption_rate: i32,
  inventory: i32,
  radius: f32,
  producer_id: i32,
}
struct Stats {
  second: i32,
  num_transactions: i32,
  num_empty_consumers: i32,
  num_total_producer_inventory: i32,
}
struct Size{
  min_x: f32,
  min_y: f32,
  max_x: f32,
  max_y: f32,
}
struct SplinePoint{
    color: vec4<f32>,
    start_pos: vec4<f32>,
    current_pos: vec4<f32>,
    end_pos: vec4<f32>,
    step_size: vec4<f32>,
    radius: f32,
    to_start: u32,
    spline_id: u32,
    point_id: u32,
    len: u32,
}
struct Point {
    color: vec4<f32>,
    position: vec4<f32>,
    radius: f32,
}
const PI: f32 = 3.14159;
@group(0) @binding(0) var<storage, read_write> consumers: array<Consumer>;
@group(0) @binding(1) var<storage, read_write> stats: vec4<i32>;
@group(0) @binding(2) var<storage, read> size: Size;
@group(0) @binding(3) var<storage, read_write> points: array<SplinePoint>;
@group(0) @binding(4) var<storage, read_write> splines: array<Point>;

fn calculateSplinePoint(t: f32, points: mat2x4<f32>) -> vec4<f32> {
    let tt = t * t;
    let ttt = tt * t;
    let q1 = -ttt + (2 * tt) - t;
    let q2 = (3 * ttt) - (5 * tt) + 2;
    let q3 = (-3 * ttt) + (4 * tt) + t;
    let q4 = ttt - tt;
    let influence = vec4(q1, q2, q3, q4);
    let result_x = dot(points[0], influence);
    let result_y = dot(points[1], influence);
    let sx = 0.5 * result_x;
    let sy = 0.5 * result_y;
    return vec4(sx, sy, 0, 0);
}
