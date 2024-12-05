const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const wgpu = zgpu.wgpu;
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Statistics = @import("statistics.zig");
const Consumer = @import("consumer.zig");
const Producer = @import("producer.zig");
const Wgpu = @import("wgpu.zig");
const Circle = @import("circle.zig");
const Callbacks = @import("callbacks.zig");

pub const Pos = struct {
    x: f32,
    y: f32,
    margin: struct {
        percent: f32 = 0.02,
        top: bool = true,
        bottom: bool = true,
        left: bool = true,
        right: bool = true,
    } = .{},
};

pub fn Slider(comptime T: type) type {
    return struct {
        min: T,
        val: T,
        prev: T,
        max: T,

        pub fn init(min: T, val: T, max: T) Slider(T) {
            return .{ .min = min, .val = val, .prev = val, .max = max };
        }
    };
}

const GuiFn = *const fn (demo: *DemoState) void;
pub const Agent = enum { consumer, producer, custom };
pub const AgentUnion = union(Agent) {
    consumer: u32,
    producer: u32,
    custom: GuiFn,
};
pub const VariableParams = struct {
    min: f64,
    mid: f64,
    max: f64,
    margin: f64,
    ratio: f64,
    variable: bool,
    extra: bool,
    title: [:0]const u8,
    help: ?[:0]const u8 = null,
    agent: AgentUnion,
};
pub fn Variable(comptime T: type) type {
    return struct {
        slider: Slider(T),
        wave: Wave,
        extra: bool,
        title: [:0]const u8,
        help: ?[:0]const u8,
        scale: [:0]const u8,
        plot_name: [:0]const u8,
        agent: AgentUnion,

        pub fn init(
            alloc: std.mem.Allocator,
            params: VariableParams,
        ) Variable(T) {
            const dist_to_min = params.mid - params.min;
            const dist_to_max = params.max - params.mid;
            const wave_amplitude = @min(dist_to_min, dist_to_max);
            const scale = params.max / 1000;
            const wave_margin = params.margin * scale;
            var wave_max = params.mid + wave_amplitude - wave_margin;
            if (wave_max <= 0) wave_max += wave_margin;

            var s_min: T = undefined;
            var s_mid: T = undefined;
            var s_max: T = undefined;

            switch (T) {
                f32, f64 => {
                    s_min = @floatCast(params.min);
                    s_mid = @floatCast(params.mid);
                    s_max = @floatCast(params.max);
                },
                else => {
                    s_min = @intFromFloat(params.min);
                    s_mid = @intFromFloat(params.mid);
                    s_max = @intFromFloat(params.max);
                },
            }

            const scale_str = std.fmt.allocPrintZ(alloc, "{d}", .{scale}) catch unreachable;
            const plot_str = std.fmt.allocPrintZ(alloc, "{s} x {d}", .{ params.title, scale }) catch unreachable;
            return .{
                .agent = params.agent,
                .plot_name = plot_str,
                .scale = scale_str,
                .slider = Slider(T).init(s_min, s_mid, s_max),
                .title = params.title,
                .help = params.help,
                .extra = params.extra,
                .wave = Wave.init(
                    alloc,
                    params.mid,
                    wave_max,
                    params.ratio,
                    scale,
                    params.variable,
                ),
            };
        }

        pub fn deinit(self: *Variable(T), alloc: std.mem.Allocator) void {
            alloc.free(self.plot_name);
            alloc.free(self.scale);
            self.wave.deinit();
        }
    };
}

pub const Wave = struct {
    active: bool,
    scale: f64,
    mid: f64,
    max: f64,
    scaled_max: f64,
    scaled_diff: f64,
    scaled_mid: f64,
    radian_ratio: f64,
    x_max: f64,
    xv: std.ArrayList(f64),
    yv: std.ArrayList(f64),

    pub fn init(
        alloc: std.mem.Allocator,
        mid: f64,
        max: f64,
        ratio: f64,
        scale: f64,
        active: bool,
    ) Wave {
        var xv = std.ArrayList(f64).init(alloc);
        var yv = std.ArrayList(f64).init(alloc);
        createXValues(&xv);
        createYValues(&yv, mid, max, ratio, scale);
        const scaled_max = max / scale;
        const scaled_mid = mid / scale;
        const scaled_diff = scaled_max - scaled_mid;
        return .{
            .active = active,
            .mid = mid,
            .max = max,
            .scaled_max = scaled_max,
            .scaled_mid = scaled_mid,
            .scaled_diff = scaled_diff,
            .scale = scale,
            .radian_ratio = ratio,
            .x_max = (std.math.pi / 2.0) / ratio,
            .xv = xv,
            .yv = yv,
        };
    }

    pub fn deinit(self: *Wave) void {
        self.xv.deinit();
        self.yv.deinit();
    }
};

pub fn update(demo: *DemoState) void {
    setupWindowPos(demo, .{ .x = 0, .y = 0 });
    setupWindowSize(demo, .{ .x = 0.25, .y = 0.75 });
    const flags = zgui.WindowFlags.no_decoration;
    if (zgui.begin("0", .{ .flags = flags })) {
        zgui.pushIntId(0);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        parameters(demo);
        zgui.popId();
    }
    zgui.end();

    var pos: Pos = .{ .x = 0, .y = 0.75, .margin = .{ .top = false } };
    var size: Pos = .{ .x = 1, .y = 0.25, .margin = .{ .top = false } };
    setupWindowPos(demo, pos);
    setupWindowSize(demo, size);
    if (zgui.begin("1", .{ .flags = flags })) {
        zgui.pushIntId(1);
        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        plots(demo);
        zgui.popId();
    }
    zgui.end();

    pos = .{ .x = 0.25, .y = 0, .margin = .{ .left = false } };
    size = .{ .x = 0.75, .y = 0.75, .margin = .{ .left = false } };
    if (demo.timeline_visible) {
        setupWindowPos(demo, pos);
        setupWindowSize(demo, size);
        if (zgui.begin("2", .{ .flags = flags })) {
            zgui.pushIntId(2);
            zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
            timeline(demo);
            zgui.popId();
        }
        zgui.end();
    }
}

pub fn updateWaves(demo: *DemoState) void {
    const sample_idx = demo.params.sample_idx;

    var it = demo.sliders.iterator();
    while (it.next()) |entry| {
        const slider = entry.value_ptr;
        if (slider.wave.active) {
            const num = slider.wave.yv.items[sample_idx];
            slider.slider.prev = slider.slider.val;
            slider.slider.val = @intFromFloat(num * slider.wave.scale);
        }
    }

    var fit = demo.f_sliders.iterator();
    while (fit.next()) |entry| {
        const slider = entry.value_ptr;
        if (slider.wave.active) {
            const num = slider.wave.yv.items[sample_idx];
            slider.slider.prev = slider.slider.val;
            slider.slider.val = @floatCast(num * slider.wave.scale);
        }
    }

    const rad = demo.sliders.get("num_consumers").?.wave.xv.items[sample_idx];
    demo.params.radian = rad;
    demo.params.sample_idx = @mod(sample_idx + 1, SAMPLE_SIZE);
}

pub fn setupWindowPos(demo: *DemoState, pos: Pos) void {
    const sd = demo.gctx.swapchain_descriptor;
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
    const margin_x = width * pos.margin.percent;
    const margin_y = height * pos.margin.percent;
    const pos_margin_pixels = @min(margin_x, margin_y);

    var x = width * pos.x;
    if (pos.margin.left) {
        x += pos_margin_pixels;
    }

    var y = height * pos.y;
    if (pos.margin.top) {
        y += pos_margin_pixels;
    }
    zgui.setNextWindowPos(.{ .x = x, .y = y });
}

pub fn setupWindowSize(demo: *DemoState, size: Pos) void {
    const sd = demo.gctx.swapchain_descriptor;
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
    const margin_x = width * size.margin.percent;
    const margin_y = height * size.margin.percent;
    const size_margin_pixels = @min(margin_x, margin_y);

    var w = width * size.x;
    if (size.margin.left) {
        w -= size_margin_pixels;
    }
    if (size.margin.right) {
        w -= size_margin_pixels;
    }

    var h = height * size.y;
    if (size.margin.top) {
        h -= size_margin_pixels;
    }
    if (size.margin.bottom) {
        h -= size_margin_pixels;
    }
    zgui.setNextWindowSize(.{ .w = w, .h = h });
}

fn agentUpdate(demo: *DemoState, agent: AgentUnion, comptime T: type, val: T) void {
    switch (agent) {
        .consumer => |offset| {
            const cp_buf = demo.buffers.data.consumer_params;
            const r = demo.gctx.lookupResource(cp_buf).?;
            demo.gctx.queue.writeBuffer(r, offset, T, &.{val});
        },
        .producer => |offset| {
            const p_buf = demo.buffers.data.producers;
            Wgpu.setObjBufField(demo.gctx, Producer, T, offset, val, p_buf);
        },
        .custom => |func| func(demo),
    }
}

pub fn slidersFromMap(
    demo: *DemoState,
    comptime T: type,
    map: *std.StringArrayHashMap(Variable(T)),
    extra: bool,
) void {
    var it = map.iterator();
    while (it.next()) |entry| {
        const slider = entry.value_ptr;
        if (slider.extra == extra) {
            infoTitle(slider.title, slider.help, &slider.wave.active);
            const s = &slider.slider;
            var buf: [100]u8 = undefined;
            const slider_str = std.fmt.bufPrintZ(&buf, "##{s}", .{slider.title}) catch unreachable;
            if (zgui.sliderScalar(slider_str, T, .{ .v = &s.val, .min = s.min, .max = s.max })) {
                agentUpdate(demo, slider.agent, T, slider.slider.val);
                slider.slider.prev = slider.slider.val;
            }
        }
    }
}

pub fn parameters(demo: *DemoState) void {
    var it = demo.sliders.iterator();
    while (it.next()) |entry| {
        const slider = entry.value_ptr;
        if (slider.slider.prev != slider.slider.val) {
            agentUpdate(demo, slider.agent, u32, slider.slider.val);
            slider.slider.prev = slider.slider.val;
        }
    }

    var f_it = demo.f_sliders.iterator();
    while (f_it.next()) |entry| {
        const slider = entry.value_ptr;
        if (slider.slider.prev != slider.slider.val) {
            agentUpdate(demo, slider.agent, f32, slider.slider.val);
            slider.slider.prev = slider.slider.val;
        }
    }

    if (zgui.beginTabBar("##tab_bar", .{})) {
        defer zgui.endTabBar();
        if (zgui.beginTabItem("Variables", .{})) {
            defer zgui.endTabItem();
            if (zgui.button("Show Timeline", .{})) {
                demo.timeline_visible = !demo.timeline_visible;
            }
            slidersFromMap(demo, u32, &demo.sliders, false);
            slidersFromMap(demo, f32, &demo.f_sliders, false);
        }

        if (zgui.beginTabItem("Extras", .{})) {
            defer zgui.endTabItem();
            slidersFromMap(demo, u32, &demo.sliders, true);
            slidersFromMap(demo, f32, &demo.f_sliders, true);
        }
    }
    buttons(demo);
}

pub fn timeline(demo: *DemoState) void {
    const size = zgui.getWindowSize();
    const margin = 15;
    const plot_size = .{ .w = size[0] - margin, .h = size[1] - margin };

    if (zgui.plot.beginPlot("", plot_size)) {
        defer zgui.plot.endPlot();

        const flags = .{ .label = "", .flags = .{ .auto_fit = true } };
        zgui.plot.setupAxis(.x1, flags);
        zgui.plot.setupAxis(.y1, flags);

        const location_flags = .{ .north = true, .east = true };
        const legend_flags = .{ .no_buttons = true };
        zgui.plot.setupLegend(location_flags, legend_flags);

        var it = demo.sliders.iterator();
        while (it.next()) |entry| {
            const slider = entry.value_ptr;
            if (slider.wave.active) {
                plotWave(&slider.wave, slider.plot_name);
                dragTopPoint(demo, &slider.wave, it.index);
                dragMidPoint(demo, &slider.wave, it.index);
            }
        }

        var f_it = demo.f_sliders.iterator();
        while (f_it.next()) |entry| {
            const slider = entry.value_ptr;
            if (slider.wave.active) {
                plotWave(&slider.wave, slider.plot_name);
                dragTopPoint(demo, &slider.wave, it.index);
                dragMidPoint(demo, &slider.wave, it.index);
            }
        }

        plotRadianLine(demo.params.radian);
    }
}

fn plotWave(wave: *Wave, plot_name: [:0]const u8) void {
    const values = .{ .xv = wave.xv.items, .yv = wave.yv.items };
    zgui.plot.plotLine(plot_name, f64, values);
}

fn plotRadianLine(radian: f64) void {
    const x1x2 = .{ radian, radian };
    const y1y2 = .{ 0, 1000 };

    //_reserved0 hides line from legend
    const line = .{ .xv = &x1x2, .yv = &y1y2, .flags = .{ ._reserved0 = true } };
    zgui.plot.plotLine("Vertical Line", f64, line);
}

fn lightenColor(color: [4]f32, amount: f32) [4]f32 {
    return .{
        color[0] + amount,
        color[1] + amount,
        color[2] + amount,
        color[3],
    };
}

fn dragTopPoint(demo: *DemoState, wave: *Wave, id: u32) void {
    var color = zgui.plot.getLastItemColor();
    color = lightenColor(color, 0.3);

    const flags = .{
        .x = &wave.x_max,
        .y = &wave.scaled_max,
        .col = color,
        .size = 4 * demo.content_scale,
    };
    if (zgui.plot.dragPoint(@intCast(id), flags)) {
        const scaled_diff = wave.scaled_max - wave.scaled_mid;
        const scaled_min = wave.scaled_mid - scaled_diff;
        const min_outside = scaled_min < 0 or scaled_min > 1000;
        const max_outside = wave.scaled_max < 0 or wave.scaled_max > 1000;
        if (min_outside or max_outside) {
            wave.scaled_max = wave.max / wave.scale;
            wave.x_max = (std.math.pi / 2.0) / wave.radian_ratio;
            return;
        }
        wave.max = wave.scaled_max * wave.scale;
        const ratio = (std.math.pi / 2.0) / wave.x_max;
        wave.radian_ratio = ratio;
        wave.scaled_diff = scaled_diff;

        createYValues(
            &wave.yv,
            wave.mid,
            wave.max,
            wave.radian_ratio,
            wave.scale,
        );
    }
}

fn dragMidPoint(demo: *DemoState, wave: *Wave, id: u32) void {
    var color = zgui.plot.getLastItemColor();
    color = lightenColor(color, 0.3);
    var zero: f64 = 0;

    const flags = .{
        .x = &zero,
        .y = &wave.scaled_mid,
        .col = color,
        .size = 4 * demo.content_scale,
    };
    if (zgui.plot.dragPoint(@intCast(id + 1000), flags)) {
        const scaled_max = wave.scaled_mid + wave.scaled_diff;
        const scaled_min = wave.scaled_mid - wave.scaled_diff;
        const min_outside = scaled_min < 0 or scaled_min > 1000;
        const max_outside = scaled_max < 0 or scaled_max > 1000;
        if (min_outside or max_outside) {
            wave.scaled_mid = wave.scaled_max - wave.scaled_diff;
            return;
        }

        wave.mid = wave.scaled_mid * wave.scale;
        wave.max = scaled_max * wave.scale;
        wave.scaled_max = scaled_max;

        createYValues(
            &wave.yv,
            wave.mid,
            wave.max,
            wave.radian_ratio,
            wave.scale,
        );
    }
}

pub fn consumerSize(demo: *DemoState) void {
    demo.buffers.vertex.circle = Circle.createVertexBuffer(
        demo.gctx,
        40,
        demo.f_sliders.get("consumer_size").?.slider.val,
    );
}

pub fn numConsumerUpdate(demo: *DemoState) void {
    const new = demo.sliders.get("num_consumers").?.slider.val;
    Statistics.setNum(demo, new, .consumers);

    const old: u32 = demo.buffers.data.consumers.mapping.num_structs;
    if (old >= new) {
        demo.buffers.data.consumers.mapping.num_structs = new;
    } else {
        Consumer.generateBulk(demo, new - old);
    }
}

pub fn numProducersUpdate(demo: *DemoState) void {
    const new = demo.sliders.get("num_producers").?.slider.val;
    Statistics.setNum(demo, new, .producers);

    const old: u32 = demo.buffers.data.producers.mapping.num_structs;
    if (old >= new) {
        demo.buffers.data.producers.mapping.num_structs = new;
    } else {
        Producer.generateBulk(demo, new - old);
    }
}

fn infoTitle(title: [:0]const u8, help: ?[:0]const u8, slider: *bool) void {
    zgui.textUnformatted(title);
    if (help) |h| {
        zgui.sameLine(.{});
        zgui.textDisabled("(?)", .{});
        if (zgui.isItemHovered(.{})) {
            _ = zgui.beginTooltip();
            zgui.textUnformatted(h);
            zgui.endTooltip();
        }
    }

    var buf: [100]u8 = undefined;
    const checkbox_str = std.fmt.bufPrintZ(&buf, "Variable##{s}", .{title}) catch unreachable;
    zgui.sameLine(.{});
    _ = zgui.checkbox(checkbox_str, .{ .v = slider });
}

fn buttons(demo: *DemoState) void {
    if (zgui.button("Start", .{})) {
        demo.running = true;
    }
    zgui.sameLine(.{});
    if (zgui.button("Stop", .{})) {
        demo.running = false;
    }
    zgui.sameLine(.{});
    if (zgui.button("Restart", .{})) {
        demo.running = true;
        Main.restartSimulation(demo);
    }

    if (zgui.button("Supply Shock", .{})) {
        Wgpu.setObjBufField(
            demo.gctx,
            Producer,
            u32,
            @offsetOf(Producer, "inventory"),
            0,
            demo.buffers.data.producers,
        );
    }

    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("Set all producer inventory to 0.");
        zgui.endTooltip();
    }
}

pub fn plots(demo: *DemoState) void {
    const size = zgui.getWindowSize();
    const margin = 15;
    const plot_size = .{ .w = size[0] - margin, .h = size[1] - margin };

    if (zgui.plot.beginPlot("", plot_size)) {
        defer zgui.plot.endPlot();

        var y_flags: zgui.plot.AxisFlags = .{ .auto_fit = true };
        if (demo.params.plot_hovered) {
            y_flags = .{ .lock_min = true };
        }

        const x_axis = .{ .label = "", .flags = .{ .auto_fit = true } };
        const y_axis = .{ .label = "", .flags = y_flags };
        zgui.plot.setupAxis(.x1, x_axis);
        zgui.plot.setupAxis(.y1, y_axis);
        zgui.plot.setupLegend(.{ .north = true, .west = true }, .{});

        demo.params.plot_hovered = zgui.plot.isPlotHovered();

        const stats = demo.stats;
        const nt = .{ .v = stats.num_transactions.items[0..] };
        const ec = .{ .v = stats.num_empty_consumers.items[0..] };
        const tpi = .{ .v = stats.num_total_producer_inventory.items[0..] };
        const acb = .{ .v = stats.avg_consumer_balance.items[0..] };
        const apb = .{ .v = stats.avg_producer_balance.items[0..] };
        const am = .{ .v = stats.avg_margin.items[0..] };

        zgui.plot.plotLineValues("Transactions", u32, nt);
        zgui.plot.plotLineValues("Empty Consumers", u32, ec);
        zgui.plot.plotLineValues("Total Producer Inventory", u32, tpi);
        zgui.plot.plotLineValues("Average Producer Balance", u32, apb);
        zgui.plot.plotLineValues("Average Consumer Balance", u32, acb);
        zgui.plot.plotLineValues("Average Producer Margin", u32, am);
    }
}

pub const RADIAN_END: f64 = 8 * std.math.pi;
pub const SAMPLE_SIZE: u32 = 2000;
pub const RADIAN_INCREMENT: f64 = RADIAN_END / @as(f64, @floatFromInt(SAMPLE_SIZE));

pub fn createXValues(xv: *std.ArrayList(f64)) void {
    xv.clearAndFree();
    var radian: f64 = 0.0;
    for (0..SAMPLE_SIZE) |_| {
        xv.append(radian) catch unreachable;
        radian += RADIAN_INCREMENT;
    }
}

pub fn createYValues(yv: *std.ArrayList(f64), mid: f64, max: f64, ratio: f64, scale: f64) void {
    yv.clearAndFree();
    var radian: f64 = 0.0;
    const diff = (max - mid);
    for (0..SAMPLE_SIZE) |_| {
        const y = mid + (diff * @sin(radian * ratio));
        yv.append(y / scale) catch unreachable;
        radian += RADIAN_INCREMENT;
    }
}
