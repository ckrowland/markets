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
    var x = position.x + vertex_position.x;
    var y = position.y + vertex_position.y;
    output.position_clip = vec4(x, y, 0.0, 1.0) * object_to_clip;
    output.color = color.xyz;
    return output;
}
