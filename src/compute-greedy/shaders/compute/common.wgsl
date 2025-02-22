struct Consumer {
  absolute_home: vec4<i32>,
  position: vec4<f32>,
  home: vec4<f32>,
  destination: vec4<f32>,
  color: vec4<f32>,
  step_size: vec2<f32>,
  radius: f32,
  inventory: u32,
  balance: u32,
  max_balance: u32,
  producer_id: i32,
  grouping_id: u32,
}

//Might need padding if something is wonky
struct ConsumerParams{
  moving_rate: f32,
  demand_rate: u32,
  income: u32,
}

struct Producer {
  params: ProducerParams,
  balance: atomic<i32>,
  inventory: atomic<i32>,
}

struct ProducerParams {
  absolute_home: vec4<i32>,
  home: vec4<f32>,
  color: vec4<f32>,
  cost: u32,
  max_inventory: u32,
  price: u32,
  margin: u32,
  prev_num_sales: u32,
  num_sales: u32,
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
