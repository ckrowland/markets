struct Consumer {
  absolute_home: vec4<i32>,
  position: vec4<f32>,
  home: vec4<f32>,
  destination: vec4<f32>,
  color: vec4<f32>,
  step_size: vec2<f32>,
  inventory: u32,
  money: u32,
  max_money: u32,
  radius: f32,
  producer_id: i32,
  grouping_id: u32,
}

struct ConsumerParams{
  moving_rate: f32,
  income: u32,
}

struct Producer {
  absolute_home: vec4<i32>,
  home: vec4<f32>,
  color: vec4<f32>,
  inventory: atomic<u32>,
  max_inventory: u32,
  money: u32,
  max_money: u32,
  price: u32,
  production_cost: u32,
}

struct Stats {
  transactions: u32,
  num_consumers: u32,
  num_producers: u32,
  num_consumer_hovers: u32,
  random_color: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> consumers: array<Consumer>;
@group(0) @binding(1) var<storage, read_write> consumer_params: ConsumerParams;
@group(0) @binding(2) var<storage, read_write> producers: array<Producer>;
@group(0) @binding(3) var<storage, read_write> stats: Stats;

fn step_sizes(pos: vec2<f32>, dest: vec2<f32>, mr: f32) -> vec2<f32>{
    let x_num_steps = num_steps(pos.x, dest.x, mr);
    let y_num_steps = num_steps(pos.y, dest.y, mr);
    let num_steps = max(x_num_steps, y_num_steps);
    let distance = dest - pos;
    return distance / num_steps;
}
fn num_steps(x: f32, y: f32, rate: f32) -> f32 {
    let distance = abs(x - y);
    if (rate > distance) { return 1.0; }
    return ceil(distance / rate);
}
