@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}
@vertex fn vs(
    @location(0) vertex_position: vec3<f32>,
    @location(1) position: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) inventory: u32,
) -> VertexOut {
    var output: VertexOut;
    let num = f32(inventory + 1000) / f32(1000);
    let scale = max(num, 1.0);
    var x = position[0] + (scale * vertex_position[0]);
    var y = position[1] + (scale * vertex_position[1]);
    output.position_clip = object_to_clip * vec4(x, y, position[2], 1.0);
    output.color = color;
    return output;
}

@fragment fn fs(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
