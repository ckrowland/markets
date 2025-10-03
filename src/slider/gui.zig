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
        settings(demo, gctx);
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
            Wgpu.getAllAsync(Producer, Callbacks.avgProducerInventory, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.producers,
                .stat_array = &demo.stats.avg_producer_inventory,
            });
            Wgpu.getAllAsync(Producer, Callbacks.avgProducerMoney, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.producers,
                .stat_array = &demo.stats.avg_producer_money,
            });
            //Wgpu.getAllAsync(Producer, Callbacks.producerMitosis, .{
            //    .gctx = demo.gctx,
            //    .obj_buf = &demo.buffers.data.producers,
            //});
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
        zgui.plot.plotLineValues("Average Producer Inventory", u32, .{
            .v = demo.stats.avg_producer_inventory.items[0..],
        });
        zgui.plot.plotLineValues("Average Producer Money", u32, .{
            .v = demo.stats.avg_producer_money.items[0..],
        });
        zgui.plot.endPlot();
    }
}

fn createSlider(
    comptime name: [:0]const u8,
    T: type,
    slider: *Main.Slider(T),
) bool {
    zgui.text(name, .{});
    return zgui.sliderScalar(
        "##" ++ name,
        T,
        .{
            .v = &slider.val,
            .min = slider.min,
            .max = slider.max,
        },
    );
}

fn settings(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
    zgui.text("  {d:.1} fps", .{demo.gctx.stats.fps});

    if (zgui.beginTabBar("##tab_bar", .{})) {
        defer zgui.endTabBar();
        if (zgui.beginTabItem("Parameters", .{})) {
            defer zgui.endTabItem();
            parameters(demo, gctx);
        }
        if (zgui.beginTabItem("Extras", .{})) {
            defer zgui.endTabItem();
            extras(demo, gctx);
        }
        zgui.dummy(.{ .w = 1.0, .h = 20.0 });
        buttons(demo);
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
        const producers = demo.buffers.data.producers;
        producers.updateU32Field(demo.gctx, 0, "inventory");
    }

    zgui.sameLine(.{});
    zgui.textDisabled("(?)", .{});
    if (zgui.isItemHovered(.{})) {
        _ = zgui.beginTooltip();
        zgui.textUnformatted("Set all producer inventory to 0.");
        zgui.endTooltip();
    }
}

fn parameters(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    const producers = demo.buffers.data.producers;

    const np = &demo.params.num_producers;
    if (createSlider("Numbers of Producers", u32, &np.slider)) {
        const resource = gctx.lookupResource(demo.stats.obj_buf.buf).?;
        gctx.queue.writeBuffer(
            resource,
            2 * @sizeOf(u32),
            u32,
            &.{np.slider.val},
        );

        if (np.old >= np.slider.val) {
            demo.buffers.data.producers.mapping.num_structs = np.slider.val;
        } else {
            Producer.generateBulk(
                gctx,
                &demo.buffers.data.producers,
                np.slider.val - np.old,
                .{
                    .max_inventory = demo.params.max_inventory.val,
                    .production_cost = demo.params.production_cost.val,
                    .price = demo.params.price.val,
                },
            );
        }
        np.old = np.slider.val;
    }

    const pc = &demo.params.production_cost;
    if (createSlider("Production Cost", u32, pc)) {
        producers.updateU32Field(demo.gctx, pc.val, "production_cost");
    }

    const price = &demo.params.price;
    if (createSlider("Price", u32, price)) {
        producers.updateU32Field(demo.gctx, price.val, "price");
    }

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    const nc = &demo.params.num_consumers;
    if (createSlider("Numbers of Consumers", u32, &nc.slider)) {
        const resource = gctx.lookupResource(demo.stats.obj_buf.buf).?;
        gctx.queue.writeBuffer(
            resource,
            @sizeOf(u32),
            u32,
            &.{nc.slider.val},
        );

        if (nc.old >= nc.slider.val) {
            demo.buffers.data.consumers.mapping.num_structs = nc.slider.val;
        } else {
            Consumer.generateBulk(
                gctx,
                demo.buffers.data.consumers.buf,
                &demo.buffers.data.consumers.mapping.num_structs,
                demo.params.aspect,
                nc.slider.val - nc.old,
            );
        }
        nc.old = nc.slider.val;
    }

    if (createSlider("Consumer Income", u32, &demo.params.income)) {
        const resource = gctx.lookupResource(demo.buffers.data.consumer_params).?;
        gctx.queue.writeBuffer(resource, 4, u32, &.{demo.params.income.val});
    }
}

fn extras(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    const producers = demo.buffers.data.producers;
    const mpi = &demo.params.max_inventory;
    if (createSlider("Max Producer Inventory", u32, mpi)) {
        producers.updateU32Field(demo.gctx, mpi.val, "max_inventory");
    }

    if (createSlider("Moving Rate", f32, &demo.params.moving_rate)) {
        const resource = gctx.lookupResource(demo.buffers.data.consumer_params).?;
        gctx.queue.writeBuffer(resource, 0, f32, &.{demo.params.moving_rate.val});
    }

    const cr = &demo.params.consumer_radius;
    if (createSlider("Consumer Size", f32, cr)) {
        demo.buffers.vertex.circle = Shapes.createCircleVertexBuffer(
            gctx,
            40,
            cr.val,
        );
    }
}
