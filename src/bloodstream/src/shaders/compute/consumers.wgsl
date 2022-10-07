@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index : u32 = global_id.x;
    var src_ball = consumers[index];
    let num_consumers = arrayLength(&consumers);
    if(index >= num_consumers) {
      return;
    }

    var velocity = src_ball.velocity;
    var position = src_ball.position;

    for (var i = 0u; i < num_consumers; i += 1u) {
        if (i == index) {
            continue;
        }
        var other_ball = consumers[i];
        let n = src_ball.position.xy - other_ball.position.xy;
        let distance = length(n);
        if (distance >= src_ball.radius + other_ball.radius) {
            continue;
        }
        //COLLISION
        let overlap = src_ball.radius + other_ball.radius - distance;
        let new_pos = src_ball.position.xy + normalize(n) * overlap/2.;
        position = vec4(new_pos.x, new_pos.y, 0.0, 0.0);

        let src_mass = pow(src_ball.radius, 2.0) * PI;
        let other_mass = pow(other_ball.radius, 2.0) * PI;
        let c = 2.*dot(n, (other_ball.velocity.xy - src_ball.velocity.xy)) / (dot(n, n) * (1./src_mass + 1./other_mass));
        let new_vel = src_ball.velocity.xy + c/src_mass * n;
        velocity = vec4(new_vel.x, new_vel.y, 0.0, 0.0);
    }

    let num_points = arrayLength(&points);
    for (var i = 0u; i < num_points; i += 1u) {
        let p = points[i];
        let first_point = p.point_id == 0;
        let second_point = p.point_id == 1;
        let last_point = p.point_id == p.len - 1;
        if (first_point || second_point || last_point) {
            continue;
        }
        let p0 = points[i - 2].current_pos;
        let p1 = points[i - 1].current_pos;
        let p2 = p.current_pos;
        let p3 = points[i + 1].current_pos;
        let diameter = f32(p.radius + 10) / distance(p1, p2);
        let num_spline_balls = 350.0;

        let points = mat2x4(p0.x, p1.x, p2.x, p3.x, p0.y, p1.y, p2.y, p3.y);
        for (var j = 0.0; j <= num_spline_balls; j += 1.0) {
            let start_t = f32(j) / num_spline_balls;
            //let n_end_t = min(1, start_t + diameter);
            let n_end_t = start_t + diameter;
            let start_edge = calculateSplinePoint(f32(start_t), points).xy;
            let end_edge = calculateSplinePoint(f32(n_end_t), points).xy;
            let radius = distance(start_edge, end_edge) / 2;
            let to_center = (end_edge - start_edge) / 2;
            let center_pos = start_edge + to_center;
            let n = src_ball.position.xy - center_pos;
            let distance = length(n);
            if (distance < radius) {
                let overlap = src_ball.radius + radius - distance;
                let new_pos = src_ball.position.xy + normalize(n) * overlap;
                position = vec4(new_pos.x, new_pos.y, 0.0, 0.0);

                let ball_mass = pow(src_ball.radius, 2.0) * PI;
                let other_mass = ball_mass;
                let other_velocity = -src_ball.velocity.xy;
                let c = 2.*dot(n, (other_velocity - src_ball.velocity.xy)) / (dot(n, n) * (1./ball_mass + 1./other_mass));
                let new_vel = src_ball.velocity.xy + c/ball_mass * n;
                velocity = vec4(new_vel.x, new_vel.y, 0.0, 0.0);
            }
        }
    }

    var vel = velocity;
    var pos = position + velocity;

    if(pos.x - src_ball.radius <= size.min_x){
        pos.x = size.min_x + src_ball.radius;
        vel.x = -vel.x;
    }
    if(pos.y - src_ball.radius <= size.min_y){
        pos.y = size.min_y + src_ball.radius;
        vel.y = -vel.y;
    }
    if(pos.x + src_ball.radius >= size.max_x){
        pos.x = size.max_x - src_ball.radius;
        vel.x = -vel.x;
    }
    if(pos.y + src_ball.radius >= size.max_y){
        pos.y = size.max_y - src_ball.radius;
        vel.y = -vel.y;
    }
    consumers[index].position = pos;
    consumers[index].velocity = vel;
}
