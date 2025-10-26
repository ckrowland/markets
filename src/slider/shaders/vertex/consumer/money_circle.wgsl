@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
@group(0) @binding(1) var<uniform> price: u32;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}
@vertex fn vs(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) vertex_position: vec3<f32>,
    @location(1) home: vec4<f32>,
    @location(2) money: u32,
) -> VertexOut {
    var output: VertexOut;
    let num = (f32(money) / f32(price)) / f32(1000);
    let scale = max(num, 1.0);
    var x = home[0] + (scale * vertex_position[0]);
    var y = home[1] + (scale * vertex_position[1]);
    output.position_clip = object_to_clip * vec4(x, y, home[2], 1.0);
    output.color = vec4(1, 1, 1, 0);
    return output;

}

@fragment fn fs(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
