@group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;

@vertex fn vs(
    @location(0) vertex_position: vec3<f32>,
    @location(1) position: vec4<f32>,
) -> @builtin(position) vec4f {
    var x = position[0] + vertex_position[0];
    var y = position[1] + vertex_position[1];
    return object_to_clip * vec4(x, y, position[2], 1.0);
}

@fragment fn fs() -> @location(0) vec4<f32> {
    return vec4f(0, 0, 1, 0);
}
