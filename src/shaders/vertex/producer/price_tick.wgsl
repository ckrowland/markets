@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}
@vertex fn vs(
    @location(0) vertex_position: vec3<f32>,
    @location(1) home: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) price: u32,
) -> VertexOut {
    var output: VertexOut;
    let scale = 2.0;
    let abs_x = 15 * abs(vertex_position[0]);
    var x = home[0] + abs_x + (scale * vertex_position[0]);
    var y = home[1] + f32(price / 50) - abs_x + (scale * vertex_position[1]);
    output.position_clip = object_to_clip * vec4(x, y, home[2], 1.0);
    output.color = color;
    return output;
}

@fragment fn fs(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
