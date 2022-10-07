@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index: u32 = global_id.x; 
    let num_points = arrayLength(&points);
    if (index >= num_points) {
        return;
    }
    let p = points[index];

    let start = points[index].start_pos;
    let current = points[index].current_pos;
    let end = points[index].end_pos;
    let at_start = distance(start, current) <= 0.5;
    let at_end = distance(end, current) <= 0.5;
    if (!at_start || !at_end) {
        if (at_end) {
            points[index].to_start = 1;
            points[index].step_size = -p.step_size;
        }
        if (at_start) {
            points[index].to_start = 0;
            points[index].step_size = -p.step_size;
        }
        var diff = end - current;
        if (points[index].to_start == 1) {
            diff = start - current;
        }
        points[index].current_pos += points[index].step_size;
    }

    //Update Spline positions
    let first_point = p.point_id == 0;
    let second_point = p.point_id == 1;
    let last_point = p.point_id == p.len - 1;
    if (first_point || second_point || last_point) {
        return;
    }
    let p0 = points[index - 2].current_pos;
    let p1 = points[index - 1].current_pos;
    let p2 = points[index + 0].current_pos;
    let p3 = points[index + 1].current_pos;
    let points = mat2x4(p0.x, p1.x, p2.x, p3.x, p0.y, p1.y, p2.y, p3.y);

    for (var j = 0u; j < 1000; j += 1) {
        let t = f32(j) * 0.001;
        let new_point = calculateSplinePoint(t, points);
        let idx = (index * 1000) + j;
        splines[idx].position = new_point;
    }
}
