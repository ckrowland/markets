@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) color: vec3<f32>,
}
@vertex fn main(
    @location(0) vertex_position: vec3<f32>,
    @location(1) position: vec4<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOut {
    var output: VertexOut;
    var x = position[0] + vertex_position[0];
    var y = position[1] + vertex_position[1];
    output.position_clip = object_to_clip * vec4(x, y, position[2], 1.0);
    output.color = color.xyz;
    return output;
}
