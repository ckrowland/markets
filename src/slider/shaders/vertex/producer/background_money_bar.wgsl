@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}
@vertex fn vs(
    @location(0) vertex_position: vec3<f32>,
    @location(1) home: vec4<f32>,
) -> VertexOut {
    var output: VertexOut;
    var x = home[0] + 50 + vertex_position[0];
    var y = home[1] - 10 + vertex_position[1];
    output.position_clip = object_to_clip * vec4(x, y, home[2], 1.0);
    output.color = vec4(0.15, 0.36, 0.04, 0);
    return output;
}

@fragment fn fs(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
