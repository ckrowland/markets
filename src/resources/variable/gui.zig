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

const gui_fn = *const fn (demo: *DemoState) void;
pub const Window = struct {
    pos: Pos,
    size: Pos,
    num_id: i32,
    str_id: [:0]const u8,
    visible: bool,
    func: gui_fn,

    const Pos = struct {
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

pub fn Variable(comptime T: type) type {
    return struct {
        slider: Slider(T),
        wave: Wave,

        pub fn init(
            alloc: std.mem.Allocator,
            min: f64,
            mid: f64,
            max: f64,
            margin: f64,
            ratio: f64,
            variable: bool,
        ) Variable(T) {
            const dist_to_min = mid - min;
            const dist_to_max = max - mid;
            const wave_amplitude = @min(dist_to_min, dist_to_max);
            const scale = max / 1000;
            const wave_margin = margin * scale;
            var wave_max = mid + wave_amplitude - wave_margin;
            if (wave_max <= 0) wave_max += wave_margin;

            var s_min: T = undefined;
            var s_mid: T = undefined;
            var s_max: T = undefined;

            switch (T) {
                f32, f64 => {
                    s_min = @floatCast(min);
                    s_mid = @floatCast(mid);
                    s_max = @floatCast(max);
                },
                else => {
                    s_min = @intFromFloat(min);
                    s_mid = @intFromFloat(mid);
                    s_max = @intFromFloat(max);
                },
            }
            return .{
                .slider = Slider(T).init(s_min, s_mid, s_max),
                .wave = Wave.init(alloc, mid, wave_max, ratio, scale, variable),
            };
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

pub const Agent = enum { consumer, producer, custom };
pub const SliderInfo = struct {
    title: [:0]const u8,
    field: [:0]const u8,
    scale: [:0]const u8 = "1",
    help: ?[:0]const u8 = null,
    agent: union(Agent) {
        consumer: bool,
        producer: bool,
        custom: gui_fn,
    },
};

pub const Variables = struct {
    num_consumers: SliderInfo = .{
        .title = "Number of Consumers",
        .scale = "10",
        .field = "num_consumers",
        .agent = .{ .custom = numConsumerUpdate },
    },
    num_producers: SliderInfo = .{
        .title = "Number of Producers",
        .scale = "0.1",
        .field = "num_producers",
        .agent = .{ .custom = numProducersUpdate },
    },
    max_demand_rate: SliderInfo = .{
        .title = "Max Demand Rate",
        .help = "The maximum amount consumers will buy from producers " ++
            "if they have enough money.",
        .field = "max_demand_rate",
        .scale = "0.3",
        .agent = .{ .consumer = true },
    },
    consumer_income: SliderInfo = .{
        .title = "Consumer Income",
        .help = "How much consumers earn each frame",
        .field = "income",
        .scale = "0.5",
        .agent = .{ .consumer = true },
    },
    consumer_size: SliderInfo = .{
        .title = "Consumer Size",
        .field = "consumer_size",
        .scale = "0.003",
        .agent = .{ .custom = consumerSize },
    },
    price: SliderInfo = .{
        .title = "Resource Price",
        .help = "The cost a consumer must pay to buy 1 resource item.",
        .field = "price",
        .scale = "1",
        .agent = .{ .producer = true },
    },
    production_rate: SliderInfo = .{
        .title = "Production Rate",
        .help = "How many resources a producer creates each cycle.",
        .field = "production_rate",
        .scale = "1",
        .agent = .{ .producer = true },
    },
    max_producer_inventory: SliderInfo = .{
        .title = "Max Producer Inventory",
        .help = "The maximum amount of resources a producer can hold",
        .field = "max_inventory",
        .scale = "10",
        .agent = .{ .producer = true },
    },
    moving_rate: SliderInfo = .{
        .title = "Moving Rate",
        .help = "How fast consumers move to and from producers",
        .field = "moving_rate",
        .scale = "0.02",
        .agent = .{ .consumer = true },
    },
};

pub fn update(demo: *DemoState) void {
    for (demo.imgui_windows) |window| {
        if (window.visible) {
            setupWindowPos(demo, window);
            setupWindowSize(demo, window);
            runWindow(demo, window);
        }
    }
}

pub fn updateWaves(demo: *DemoState) void {
    const sample_idx = demo.params.sample_idx;
    inline for (@typeInfo(Variables).Struct.fields) |f| {
        const info = @field(demo.sliders, f.name);
        const param = &@field(demo.params, info.field);
        if (param.wave.active) {
            const num = param.wave.yv.items[sample_idx];

            param.slider.prev = param.slider.val;
            if (@TypeOf(param.slider.val) == f32) {
                param.slider.val = @floatCast(num * param.wave.scale);
            } else {
                param.slider.val = @intFromFloat(num * param.wave.scale);
            }
        }
    }

    const rad = demo.params.num_consumers.wave.xv.items[sample_idx];
    demo.params.radian = rad;

    demo.params.sample_idx = @mod(sample_idx + 1, SAMPLE_SIZE);
}

pub fn setupWindowPos(demo: *DemoState, window: Window) void {
    const sd = demo.gctx.swapchain_descriptor;
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
    const pos = window.pos;
    const pos_margin_pixels = getMarginPixels(sd, pos.margin.percent);

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

pub fn setupWindowSize(demo: *DemoState, window: Window) void {
    const sd = demo.gctx.swapchain_descriptor;
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));

    const size = window.size;
    const size_margin_pixels = getMarginPixels(sd, size.margin.percent);

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

pub fn runWindow(demo: *DemoState, window: Window) void {
    const flags = zgui.WindowFlags.no_decoration;
    const start = zgui.begin(window.str_id, .{ .flags = flags });
    defer zgui.end();

    if (start) {
        zgui.pushIntId(window.num_id);
        defer zgui.popId();

        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        window.func(demo);
    }
}

fn getMarginPixels(sd: wgpu.SwapChainDescriptor, margin_percent: f32) f32 {
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
    const margin_x = width * margin_percent;
    const margin_y = height * margin_percent;
    return @min(margin_x, margin_y);
}

pub fn parameters(demo: *DemoState) void {
    inline for (@typeInfo(Variables).Struct.fields) |f| {
        const info = @field(demo.sliders, f.name);
        const param = &@field(demo.params, info.field);
        if (param.wave.active) {
            const slider_type = @TypeOf(param.slider.val);
            checkIfSliderUpdated(demo, slider_type, info);
        }
    }
    if (zgui.beginTabBar("##tab_bar", .{})) {
        defer zgui.endTabBar();
        if (zgui.beginTabItem("Variables", .{})) {
            defer zgui.endTabItem();
            showTimeline(demo);
            inline for (@typeInfo(Variables).Struct.fields) |f| {
                const info = @field(demo.sliders, f.name);
                const param = &@field(demo.params, info.field);
                if (param.wave.active) {
                    infoTitle(demo, info);
                    const slider_type = @TypeOf(param.slider.val);
                    displaySlider(demo, slider_type, &param.slider, info);
                }
            }
        }

        if (zgui.beginTabItem("Constants", .{})) {
            defer zgui.endTabItem();
            showTimeline(demo);
            inline for (@typeInfo(Variables).Struct.fields) |f| {
                const info = @field(demo.sliders, f.name);
                const param = &@field(demo.params, info.field);
                if (!param.wave.active) {
                    infoTitle(demo, info);
                    const slider_type = @TypeOf(param.slider.val);
                    displaySlider(demo, slider_type, &param.slider, info);
                }
            }
        }
    }
    buttons(demo);
}

fn showTimeline(demo: *DemoState) void {
    if (zgui.button("Show Timeline", .{})) {
        demo.imgui_windows[2].visible = !demo.imgui_windows[2].visible;
    }
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

        inline for (@typeInfo(Variables).Struct.fields, 0..) |f, i| {
            const info = @field(demo.sliders, f.name);
            const param = &@field(demo.params, info.field);
            if (param.wave.active) {
                plotWave(&param.wave, info);
                dragTopPoint(demo, info, i);
                dragMidPoint(demo, info, i);
            }
        }

        plotRadianLine(demo.params.radian);
    }
}

fn plotWave(wave: *Wave, comptime info: SliderInfo) void {
    const values = .{ .xv = wave.xv.items, .yv = wave.yv.items };
    zgui.plot.plotLine(info.title ++ " x " ++ info.scale, f64, values);
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

fn dragTopPoint(demo: *DemoState, comptime info: SliderInfo, id: i32) void {
    const param = &@field(demo.params, info.field);

    var color = zgui.plot.getLastItemColor();
    color = lightenColor(color, 0.3);

    const wave = &param.wave;
    const flags = .{
        .x = &wave.x_max,
        .y = &wave.scaled_max,
        .col = color,
        .size = 4 * demo.content_scale,
    };
    if (zgui.plot.dragPoint(id, flags)) {
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

fn dragMidPoint(demo: *DemoState, comptime info: SliderInfo, id: i32) void {
    const param = &@field(demo.params, info.field);

    var color = zgui.plot.getLastItemColor();
    color = lightenColor(color, 0.3);
    var zero: f64 = 0;

    const wave = &param.wave;
    const flags = .{
        .x = &zero,
        .y = &wave.scaled_mid,
        .col = color,
        .size = 4 * demo.content_scale,
    };
    if (zgui.plot.dragPoint(id + 1000, flags)) {
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

fn checkIfSliderUpdated(demo: *DemoState, comptime T: type, comptime info: SliderInfo) void {
    const param = &@field(demo.params, info.field);
    const value_changed = param.slider.prev != param.slider.val;
    if (value_changed) {
        switch (info.agent) {
            .consumer => {
                const offset: u32 = @intCast(std.meta.fieldIndex(Consumer.Params, info.field).?);
                demo.gctx.queue.writeBuffer(
                    demo.gctx.lookupResource(demo.buffers.data.consumer_params).?,
                    offset * @sizeOf(u32),
                    T,
                    &.{param.slider.val},
                );
            },
            .producer => Wgpu.setObjBufField(
                demo,
                Producer,
                @TypeOf(param.slider.val),
                info.field,
                param.slider.val,
            ),
            .custom => |func| func(demo),
        }
        param.slider.prev = param.slider.val;
    }
}

fn displaySlider(
    demo: *DemoState,
    comptime T: type,
    slider: *Slider(T),
    comptime info: SliderInfo,
) void {
    const flags = .{ .v = &slider.val, .min = slider.min, .max = slider.max };
    const slider_changed = zgui.sliderScalar("##" ++ info.title, T, flags);
    if (slider_changed) {
        switch (info.agent) {
            .consumer => {
                const offset: u32 = @intCast(std.meta.fieldIndex(Consumer.Params, info.field).?);
                demo.gctx.queue.writeBuffer(
                    demo.gctx.lookupResource(demo.buffers.data.consumer_params).?,
                    offset * @sizeOf(u32),
                    T,
                    &.{slider.val},
                );
            },
            .producer => Wgpu.setObjBufField(demo, Producer, T, info.field, slider.val),
            .custom => |func| func(demo),
        }
        slider.prev = slider.val;
    }
}

fn consumerSize(demo: *DemoState) void {
    demo.buffers.vertex.circle = Circle.createVertexBuffer(
        demo.gctx,
        40,
        demo.params.consumer_size.slider.val,
    );
}

fn numConsumers(demo: *DemoState) void {
    zgui.text("Number Of Consumers", .{});
    const param = &demo.params.num_consumers;
    const new: *u32 = @ptrCast(&param.val);
    const flags = .{ .v = new, .min = param.min, .max = param.max };
    _ = zgui.sliderScalar("##nc", u32, flags);
    numConsumerUpdate(demo);
}

pub fn numConsumerUpdate(demo: *DemoState) void {
    const new = demo.params.num_consumers.slider.val;
    Statistics.setNum(demo, new, .consumers);

    const old: u32 = demo.buffers.data.consumers.mapping.num_structs;
    if (old >= new) {
        const buf = demo.buffers.data.consumers.buf;
        Wgpu.shrinkBuffer(demo.gctx, buf, Consumer, new);
        demo.buffers.data.consumers.mapping.num_structs = new;
    } else {
        Consumer.generateBulk(demo, new - old);
    }
}

fn numProducersUpdate(demo: *DemoState) void {
    const new = demo.params.num_producers.slider.val;
    Statistics.setNum(demo, new, .producers);

    const old: u32 = demo.buffers.data.producers.mapping.num_structs;
    if (old >= new) {
        const buf = demo.buffers.data.producers.buf;
        Wgpu.shrinkBuffer(demo.gctx, buf, Producer, new);
        demo.buffers.data.producers.mapping.num_structs = new;
    } else {
        Producer.generateBulk(demo, new - old);
    }
}

fn infoTitle(demo: *DemoState, comptime slide: SliderInfo) void {
    zgui.text(slide.title, .{});
    if (slide.help) |help| {
        zgui.sameLine(.{});
        zgui.textDisabled("(?)", .{});
        if (zgui.isItemHovered(.{})) {
            _ = zgui.beginTooltip();
            zgui.textUnformatted(help);
            zgui.endTooltip();
        }
    }
    zgui.sameLine(.{});
    if (zgui.button("Switch" ++ "##" ++ slide.field, .{})) {
        const param = &@field(demo.params, slide.field);
        param.wave.active = !param.wave.active;
    }
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
        Wgpu.setObjBufField(demo, Producer, i32, "inventory", 0);
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

        zgui.plot.plotLineValues("Transactions", u32, nt);
        zgui.plot.plotLineValues("Empty Consumers", u32, ec);
        zgui.plot.plotLineValues("Total Producer Inventory", u32, tpi);
        zgui.plot.plotLineValues("Average Consumer Balance", u32, acb);
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
