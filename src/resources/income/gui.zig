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

pub const FullPos = struct {
    pos: Pos,
    size: Pos,
    id: GuiID,
};

pub const GuiID = struct {
    num: i32,
    str: [:0]const u8,
};

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

pub fn update(demo: *DemoState) void {
    setUpWindow(demo, parameters, demo.imgui_windows[0]);
    setUpWindow(demo, plots, demo.imgui_windows[1]);
}

const gui_fn = *const fn (demo: *DemoState) void;
pub fn setUpWindow(demo: *DemoState, func: gui_fn, full_pos: FullPos) void {
    setupWindowPos(demo, full_pos.pos);
    setupWindowSize(demo, full_pos.size);
    runWindow(demo, func, full_pos.id);
}

pub fn setupWindowPos(demo: *DemoState, pos: Pos) void {
    const sd = demo.gctx.swapchain_descriptor;
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
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

pub fn setupWindowSize(demo: *DemoState, size: Pos) void {
    const sd = demo.gctx.swapchain_descriptor;
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
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

pub fn runWindow(demo: *DemoState, func: gui_fn, id: GuiID) void {
    const flags = zgui.WindowFlags.no_decoration;
    const start = zgui.begin(id.str, .{ .flags = flags });
    defer zgui.end();

    if (start) {
        zgui.pushIntId(id.num);
        defer zgui.popId();

        zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
        func(demo);
    }
}

fn getMarginPixels(sd: wgpu.SwapChainDescriptor, margin_percent: f32) f32 {
    const width = @as(f32, @floatFromInt(sd.width));
    const height = @as(f32, @floatFromInt(sd.height));
    const margin_x = width * margin_percent;
    const margin_y = height * margin_percent;
    return @min(margin_x, margin_y);
}

fn parameters(demo: *DemoState) void {
    income_quartiles(demo);
    max_demand_rate(demo);
    moving_rate(demo);
    num_producers(demo);
    production_rate(demo);
    buttons(demo);

    // zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    max_producer_inventory(demo);
    consumer_size(demo);
}

fn income_quartiles(demo: *DemoState) void {
    const plot_flags = .{ .flags = zgui.plot.Flags.canvas_only };
    if (zgui.plot.beginPlot("Consumer Income", plot_flags)) {
        const x_flags = .{ .label = "Income Rate", .flags = .{} };
        const y_flags = .{ .label = "Number of Consumers", .flags = .{} };
        const x_limit_flags = .{ .min = 0, .max = 0.5, .cond = .always };
        const y_limit_flags = .{ .min = 0, .max = 5000, .cond = .always };

        zgui.plot.setupAxis(.x1, x_flags);
        zgui.plot.setupAxis(.y1, y_flags);
        zgui.plot.setupAxisLimits(.x1, x_limit_flags);
        zgui.plot.setupAxisLimits(.y1, y_limit_flags);

        const income_ptr = &demo.params.consumer_incomes;
        inline for (income_ptr, 0..) |_, i| {
            const ptr = &income_ptr[i].new;
            zgui.plot.plotBars("Consumer Income Brackets", f64, .{
                .xv = &.{ptr.income},
                .yv = &.{ptr.num},
                .bar_size = 0.05,
            });
            const red = .{ 1, 0, 0, 1 };
            const flags = .{ .x = &ptr.income, .y = &ptr.num, .col = red };
            if (zgui.plot.dragPoint(i, flags)) {
                if (ptr.num <= 0) ptr.num = 0.0001;
                if (ptr.income > 0.5) ptr.income = 0.5;
                if (ptr.income < 0) ptr.income = 0;

                const point = &demo.params.consumer_incomes[i];
                const new: u32 = @intFromFloat(point.new.num);
                const old: u32 = @intFromFloat(point.old.num);

                if (new > old) {
                    Consumer.expandQuartile(demo, i, new - old);
                } else if (new < old) {
                    Consumer.shrinkQuartile(demo, i, old - new);
                }

                if (point.new.income != point.old.income) {
                    const f_income: f32 = @floatCast(point.new.income);
                    Consumer.setQuartileIncome(demo, i, f_income);
                }

                point.old.income = point.new.income;
                point.old.num = point.new.num;
            }
        }
        zgui.plot.endPlot();
    }
}

fn max_demand_rate(demo: *DemoState) void {
    zgui.text("Max Demand Rate", .{});
    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted(
            "How much consumers take from producers if they have enough money.",
        );
        zgui.endTooltip();
    }

    const mdr = &demo.params.max_demand_rate;
    const flags = .{ .v = mdr, .min = 1, .max = 200 };
    if (zgui.sliderScalar("##dr", u32, flags)) {
        Consumer.setParamAll(demo, "max_demand_rate", u32, mdr.*);
    }
}

fn moving_rate(demo: *DemoState) void {
    zgui.text("Moving Rate", .{});
    const mr = &demo.params.moving_rate;
    const flags = .{ .v = mr, .min = 1.0, .max = 20 };
    if (zgui.sliderScalar("##mr", f32, flags)) {
        Consumer.setParamAll(demo, "moving_rate", f32, mr.*);
    }
}

fn consumer_size(demo: *DemoState) void {
    zgui.text("Consumer Size", .{});
    const cr = &demo.params.consumer_radius;
    const flags = .{ .v = cr, .min = 1, .max = 3 };
    if (zgui.sliderScalar("##cs", f32, flags)) {
        demo.buffers.vertex.circle = Circle.createVertexBuffer(
            demo.gctx,
            40,
            cr.*,
        );
    }
}

fn num_producers(demo: *DemoState) void {
    zgui.text("Number Of Producers", .{});
    const new = &demo.params.num_producers;
    const old: u32 = @intCast(demo.buffers.data.producers.list.items.len);
    const flags = .{ .v = new, .min = 1, .max = 100 };
    if (zgui.sliderScalar("##np", u32, flags)) {
        Statistics.setNum(demo, new.*, .producers);

        if (old >= new.*) {
            const buf = demo.buffers.data.producers.buf;
            Wgpu.shrinkBuffer(demo.gctx, buf, Producer, new.*);
            demo.buffers.data.producers.list.resize(new.*) catch unreachable;
            demo.buffers.data.producers.mapping.num_structs = new.*;
        } else {
            Producer.generateBulk(demo, new.* - old);
        }
    }
}

fn production_rate(demo: *DemoState) void {
    zgui.text("Production Rate", .{});
    const pr = &demo.params.production_rate;
    const flags = .{ .v = pr, .min = 1, .max = 1000 };
    if (zgui.sliderScalar("##pr", u32, flags)) {
        Producer.setParamAll(demo, "production_rate", u32, pr.*);
    }
}

fn max_producer_inventory(demo: *DemoState) void {
    zgui.text("Max Producer Inventory", .{});
    const mi = &demo.params.max_inventory;
    const flags = .{ .v = mi, .min = 10, .max = 10000 };
    if (zgui.sliderScalar("##mi", u32, flags)) {
        Producer.setParamAll(demo, "max_inventory", u32, mi.*);
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
        Producer.setParamAll(demo, "inventory", i32, 0);
    }

    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("Set all producer inventory to 0.");
        zgui.endTooltip();
    }
}

fn plots(demo: *DemoState) void {
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
