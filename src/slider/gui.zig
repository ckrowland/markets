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
    const sd = demo.gctx.swapchain_descriptor;
    Gui.setupWindowPos(sd, .{ .x = 0, .y = 0 });
    Gui.setupWindowSize(sd, .{ .x = 0.2, .y = 1 });
    const flags = zgui.WindowFlags.no_decoration;
    if (zgui.begin("0", .{ .flags = flags })) {
        defer zgui.end();
        zgui.pushIntId(2);
        settings(demo, demo.gctx);
        zgui.popId();
    }

    //zgui.plot.showDemoWindow(null);
    Gui.setupWindowPos(sd, .{ .x = 0.2, .y = 0, .margin = .{ .left = false } });
    Gui.setupWindowSize(sd, .{ .x = 0.8, .y = 1, .margin = .{ .right = false } });
    if (demo.stats_page.val and demo.stats_page.change) {
        demo.stats_page.change = false;
    }
    if (demo.stats_page.val) {
        //zgui.setNextWindowBgAlpha(.{ .alpha = 0 });
        //zgui.pushStyleColor1u(.{ .idx = .frame_bg, .c = 0 });
        //zgui.pushStyleColor1u(.{ .idx = .window_bg, .c = 0 });
        //defer zgui.popStyleColor(.{});
        //defer zgui.popStyleColor(.{});
        if (zgui.begin("Statistics", .{ .popen = &demo.stats_page.val })) {
            zgui.pushIntId(3);
            plots(demo);
            zgui.popId();
        }
        defer zgui.end();
    }

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
        const current_time = @as(f32, @floatCast(demo.gctx.stats.time));
        const seconds_passed = current_time - demo.stats.second;
        if (seconds_passed >= 1) {
            demo.stats.second = current_time;
            Wgpu.getAllAsync(Producer, Callbacks.price, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.producers,
                .stat_array = &demo.stats.price,
            });
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
            Wgpu.getAllAsync(Consumer, Callbacks.avgConsumerInventory, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.consumers,
                .stat_array = &demo.stats.avg_consumer_inventory,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.avgConsumerMoney, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.data.consumers,
                .stat_array = &demo.stats.avg_consumer_money,
            });
        }
    }
}

fn plots(demo: *DemoState) void {
    if (zgui.button("Hide Statistics", .{})) {
        demo.stats_page.val = !demo.stats_page.val;
        if (demo.stats_page.val) {
            demo.stats_page.change = true;
        }
    }

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
    zgui.sameLine(.{});
    if (zgui.button("Supply Shock", .{})) {
        const producers = demo.buffers.data.producers;
        producers.updateU32Field(demo.gctx, 0, "inventory");
    }

    if (zgui.beginTable("Stats", .{ .column = 3 })) {
        defer zgui.endTable();

        zgui.tableSetupColumn("Statistic", .{ .flags = .{ .width_fixed = true }, .init_width_or_height = 150 });
        zgui.tableSetupColumn("Value", .{ .flags = .{ .width_fixed = true }, .init_width_or_height = 60 });
        zgui.tableSetupColumn("Plot", .{});
        zgui.tableHeadersRow();
        plotRow("Price", &demo.stats.price);
        plotRow("Num Transactions", &demo.stats.num_transactions);
        plotRow("Empty Consumers", &demo.stats.num_empty_consumers);
        plotRow("Avg. Producer Inventory", &demo.stats.avg_producer_inventory);
        plotRow("Avg. Producer Money", &demo.stats.avg_producer_money);
        plotRow("Avg. Consumer Inventory", &demo.stats.avg_consumer_inventory);
        plotRow("Avg. Consumer Money", &demo.stats.avg_consumer_money);
    }
}

fn plotRow(comptime str: [:0]const u8, arr: *std.ArrayList(u32)) void {
    zgui.tableNextRow(.{});
    _ = zgui.tableSetColumnIndex(0);
    zgui.text(str, .{});
    _ = zgui.tableSetColumnIndex(1);
    zgui.text("{any}", .{arr.getLastOrNull()});
    _ = zgui.tableSetColumnIndex(2);

    if (zgui.plot.beginPlot("##" ++ str, .{
        .h = 150,
    })) {
        defer zgui.plot.endPlot();
        zgui.plot.setupAxis(.x1, .{
            .label = "",
            .flags = .{ .auto_fit = true, .no_label = true, .no_tick_labels = true },
        });
        zgui.plot.setupAxis(.y1, .{
            .label = "",
            .flags = .{ .auto_fit = true },
        });
        zgui.plot.plotLineValues(str, u32, .{
            .v = arr.items[0..],
        });
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
            extras(demo, gctx);
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

    if (zgui.button("Statistics Page", .{})) {
        demo.stats_page.val = !demo.stats_page.val;
        if (demo.stats_page.val) {
            demo.stats_page.change = true;
        }
    }
}

fn parameters(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    const producers = demo.buffers.data.producers;

    zgui.bulletText("Producer Settings", .{});
    const np = &demo.params.num_producers;
    if (createSlider("Number of Producers", u32, &np.slider)) {
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
                    .max_money = demo.params.max_producer_money.val,
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
    if (createSlider("Price Sold", u32, price)) {
        producers.updateU32Field(demo.gctx, price.val, "price");
    }

    const mpm = &demo.params.max_producer_money;
    if (createSlider("Max Producer Money", u32, mpm)) {
        producers.updateU32Field(demo.gctx, mpm.val, "max_money");
    }
    const mpi = &demo.params.max_inventory;
    if (createSlider("Max Inventory", u32, mpi)) {
        producers.updateU32Field(demo.gctx, mpi.val, "max_inventory");
    }

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    zgui.bulletText("Consumer Settings", .{});
    const nc = &demo.params.num_consumers;
    if (createSlider("Number of Consumers", u32, &nc.slider)) {
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
                demo.gctx,
                &demo.buffers.data.consumers,
                nc.slider.val - nc.old,
                .{
                    .income = demo.params.income.val,
                    .moving_rate = demo.params.moving_rate.val,
                    .max_money = demo.params.max_consumer_money.val,
                },
            );
        }
        nc.old = nc.slider.val;
    }

    const consumers = demo.buffers.data.consumers;
    const ci = &demo.params.income;
    if (createSlider("Income", u32, &demo.params.income)) {
        consumers.updateU32Field(demo.gctx, ci.val, "income");
    }

    const mcm = &demo.params.max_consumer_money;
    if (createSlider("Max Consumer Money", u32, mcm)) {
        consumers.updateU32Field(demo.gctx, mcm.val, "max_money");
    }

    const mr = &demo.params.moving_rate;
    if (createSlider("Moving Rate", f32, mr)) {
        consumers.updateF32Field(demo.gctx, mr.val, "moving_rate");
    }
}

fn extras(demo: *DemoState, gctx: *zgpu.GraphicsContext) void {
    const cs = &demo.params.consumer_size;
    if (createSlider("Agent Size", f32, cs)) {
        demo.buffers.vertex.circle = Shapes.createCircleVertexBuffer(
            gctx,
            Main.NUM_CONSUMER_SIDES,
            cs.val,
        );
        demo.params.producer_size.val = demo.params.consumer_size.val;
        demo.buffers.vertex.square = Shapes.createSquareVertexBuffer(
            gctx,
            demo.params.producer_size.val,
        );
    }
}
