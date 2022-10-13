@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) color: vec3<f32>,
}
@vertex fn main(
    @location(0) vertex_position: vec3<f32>,
    @location(1) position: vec4<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
) -> VertexOut {
    var output: VertexOut;
    var nx = vertex_position.x;
    var ny = vertex_position.y;
    let rad = 0.0;
    if (vertex_position.x > 0) {
        nx += radius;
    } else {
        nx -= radius; 
    }
    if (vertex_position.y > 0) {
        ny += radius;
    } else {
        ny -= radius; 
    }
    var x = position.x + cos(rad) * nx - sin(rad) * ny;
    var y = position.y + sin(rad) * nx + cos(rad) * ny;
    output.position_clip = vec4(x, y, 0.0, 1.0) * object_to_clip;
    output.color = color.xyz;
    return output;
}
