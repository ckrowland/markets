const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const Shapes = @import("shapes");
const Gui = @import("gui");
const Main = @import("main.zig");
const DemoState = Main.DemoState;
const Consumer = @import("consumer");
const Producer = @import("producer");
const Wgpu = @import("wgpu");
const Callbacks = @import("callbacks.zig");

pub fn update(demo: *DemoState) void {
    const gctx = demo.gctx;
    const sd = demo.gctx.swapchain_descriptor;

    var pos = Gui.setupWindowPos(sd, .{ .x = 0, .y = 0 });
    var size = Gui.setupWindowSize(sd, .{ .x = 0.25, .y = 0.75 });
    zgui.setNextWindowPos(.{ .x = pos[0], .y = pos[1] });
    zgui.setNextWindowSize(.{ .w = size[0], .h = size[1] });

    const flags = zgui.WindowFlags.no_decoration;
    if (zgui.begin("0", .{ .flags = flags })) {
        zgui.pushIntId(2);
        parameters(demo, gctx);
        zgui.popId();
    }
    zgui.end();

    pos = Gui.setupWindowPos(sd, .{ .x = 0, .y = 0.75, .margin = .{ .top = false } });
    size = Gui.setupWindowSize(sd, .{ .x = 1, .y = 0.25, .margin = .{ .top = false } });
    zgui.setNextWindowPos(.{ .x = pos[0], .y = pos[1] });
    zgui.setNextWindowSize(.{ .w = size[0], .h = size[1] });
    if (zgui.begin("1", .{ .flags = flags })) {
        zgui.pushIntId(3);
        plots(demo);
        zgui.popId();
    }
    zgui.end();

    Wgpu.checkObjBufState(u32, &demo.stats.obj_buf.mapping);
    Wgpu.checkObjBufState(Producer, &demo.buffers.data.producers.mapping);
    Wgpu.checkObjBufState(Consumer, &demo.buffers.data.consumers.mapping);

    if (demo.running) {
        //gctx.queue.writeBuffer(
        //    gctx.lookupResource(demo.stats.obj_buf.buf).?,
        //    3 * @sizeOf(u32),
        //    f32,
        //    &.{ random.float(f32), random.float(f32), random.float(f32) },
        //);
        const current_time = @as(f32, @floatCast(gctx.stats.time));
        const seconds_passed = current_time - demo.stats.second;
        if (seconds_passed >= 1) {
            demo.stats.second = current_time;
            Wgpu.getAllAsync(u32, Callbacks.numTransactions, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.stats.obj_buf,
                .stat_array = &demo.stats.num_transactions,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.emptyConsumers, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.consumers,
                .stat_array = &demo.stats.num_empty_consumers,
            });
            Wgpu.getAllAsync(Producer, Callbacks.totalInventory, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.producers,
                .stat_array = &demo.stats.num_total_producer_inventory,
            });
        }
    }
}

fn plots(demo: *DemoState) void {
    const window_size = zgui.getWindowSize();
    const margin = 15;
    const plot_width = window_size[0] - margin;
    const plot_height = window_size[1] - margin;

    if (zgui.plot.beginPlot("", .{
        .w = plot_width,
        .h = plot_height,
        .flags = .{},
    })) {
        zgui.plot.setupAxis(.x1, .{
            .label = "",
            .flags = .{ .auto_fit = true },
        });
        zgui.plot.setupAxis(.y1, .{
            .label = "",
            .flags = .{ .auto_fit = true },
        });
        zgui.plot.setupLegend(.{ .north = true, .west = true }, .{});
        zgui.plot.plotLineValues("Transactions", u32, .{
            .v = demo.stats.num_transactions.items[0..],
        });
        zgui.plot.plotLineValues("Empty Consumers", u32, .{
            .v = demo.stats.num_empty_consumers.items[0..],
        });
        zgui.plot.plotLineValues("Total Producer Inventory", u32, .{
            .v = demo.stats.num_total_producer_inventory.items[0..],
        });
        zgui.plot.endPlot();
    }
}

fn parameters(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
    zgui.bulletText(
        "{d:.3} ms/frame ({d:.1} fps)",
        .{ demo.gctx.stats.average_cpu_time, demo.gctx.stats.fps },
    );

    zgui.text("Number Of Producers", .{});

    if (zgui.sliderScalar(
        "##np",
        u32,
        .{ .v = &demo.params.num_producers.new, .min = 1, .max = 100 },
    )) {
        const num_producers = demo.params.num_producers;

        const resource = gctx.lookupResource(demo.stats.obj_buf.buf).?;
        gctx.queue.writeBuffer(
            resource,
            2 * @sizeOf(u32),
            u32,
            &.{num_producers.new},
        );

        if (num_producers.old >= num_producers.new) {
            demo.buffers.data.producers.mapping.num_structs = num_producers.new;
        } else {
            Producer.generateBulk(
                gctx,
                demo.buffers.data.producers.buf,
                &demo.buffers.data.producers.mapping.num_structs,
                demo.params.aspect,
                num_producers.new - num_producers.old,
                demo.params.production_rate,
                demo.params.max_inventory,
            );
        }
        demo.params.num_producers.old = demo.params.num_producers.new;
    }

    zgui.text("Production Rate", .{});
    if (zgui.sliderScalar(
        "##pr",
        u32,
        .{ .v = &demo.params.production_rate, .min = 1, .max = 1000 },
    )) {
        for (0..demo.buffers.data.producers.mapping.num_structs) |i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.producers.buf).?,
                i * @sizeOf(Producer) + @offsetOf(Producer, "production_rate"),
                u32,
                &.{demo.params.production_rate},
            );
        }
    }

    zgui.text("Demand Rate", .{});
    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("How much consumers take from producers on a trip.");
        zgui.endTooltip();
    }

    if (zgui.sliderScalar(
        "##dr",
        i32,
        .{ .v = &demo.params.demand_rate, .min = 1, .max = 1000 },
    )) {
        const resource = gctx.lookupResource(demo.buffers.data.consumer_params).?;
        gctx.queue.writeBuffer(resource, @sizeOf(i32), i32, &.{demo.params.demand_rate});
    }

    zgui.text("Max Producer Inventory", .{});
    if (zgui.sliderScalar("##mi", u32, .{
        .v = &demo.params.max_inventory,
        .min = 10,
        .max = 10000,
    })) {
        for (0..demo.buffers.data.producers.mapping.num_structs) |i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.producers.buf).?,
                i * @sizeOf(Producer) + @offsetOf(Producer, "max_inventory"),
                u32,
                &.{demo.params.max_inventory},
            );
        }
    }

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    zgui.text("Number of Consumers", .{});
    if (zgui.sliderScalar("##nc", u32, .{
        .v = &demo.params.num_consumers.new,
        .min = 1,
        .max = 10000,
    })) {
        const num_consumers = demo.params.num_consumers;

        const resource = gctx.lookupResource(demo.stats.obj_buf.buf).?;
        gctx.queue.writeBuffer(
            resource,
            @sizeOf(u32),
            u32,
            &.{num_consumers.new},
        );

        if (num_consumers.old >= num_consumers.new) {
            demo.buffers.data.consumers.mapping.num_structs = num_consumers.new;
        } else {
            Consumer.generateBulk(
                gctx,
                demo.buffers.data.consumers.buf,
                &demo.buffers.data.consumers.mapping.num_structs,
                demo.params.aspect,
                num_consumers.new - num_consumers.old,
            );
        }
        demo.params.num_consumers.old = demo.params.num_consumers.new;
    }

    zgui.text("Consumer Income", .{});
    if (zgui.sliderScalar("##i", u32, .{
        .v = &demo.params.income,
        .min = 0,
        .max = 20,
    })) {
        const resource = gctx.lookupResource(demo.buffers.data.consumer_params).?;
        gctx.queue.writeBuffer(resource, 8, u32, &.{demo.params.income});
    }

    zgui.text("Moving Rate", .{});
    if (zgui.sliderScalar("##mr", f32, .{
        .v = &demo.params.moving_rate,
        .min = 1.0,
        .max = 20,
    })) {
        const resource = gctx.lookupResource(demo.buffers.data.consumer_params).?;
        gctx.queue.writeBuffer(resource, 0, f32, &.{demo.params.moving_rate});
    }

    zgui.text("Consumer Size", .{});
    if (zgui.sliderScalar("##cs", f32, .{
        .v = &demo.params.consumer_radius,
        .min = 1,
        .max = 12,
    })) {
        demo.buffers.vertex.circle = Shapes.createCircleVertexBuffer(
            gctx,
            40,
            demo.params.consumer_radius,
        );
    }

    if (zgui.button("Start", .{})) {
        demo.running = true;
    }
    if (zgui.button("Stop", .{})) {
        demo.running = false;
    }
    if (zgui.button("Restart", .{})) {
        demo.running = true;
        Main.restartSimulation(demo);
    }
    if (zgui.button("Supply Shock", .{})) {
        for (0..demo.buffers.data.producers.mapping.num_structs) |i| {
            gctx.queue.writeBuffer(
                gctx.lookupResource(demo.buffers.data.producers.buf).?,
                i * @sizeOf(Producer) + @offsetOf(Producer, "inventory"),
                i32,
                &.{0},
            );
        }
    }

    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("Set all producer inventory to 0.");
        zgui.endTooltip();
    }
}
