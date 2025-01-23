@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
@group(0) @binding(1) var<uniform> colors: array<vec4<f32>, 25>;

struct VertexOutput{
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) grouping_id: u32,
};

@vertex fn vs(
    @location(0) vertex_position: vec3<f32>,
    @location(1) position: vec4<f32>,
    @location(2) grouping_id: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var x = position[0] + vertex_position[0];
    var y = position[1] + vertex_position[1];
    out.position = object_to_clip * vec4(x, y, position[2], 1.0);
    out.grouping_id = grouping_id;
    return out;
}

@fragment fn fs(in: VertexOutput) -> @location(0) vec4<f32> {
    return colors[in.grouping_id];
}
