const std = @import("std");
const random = std.crypto.random;
const zgpu = @import("zgpu");
const zgui = @import("zgui");
const Callbacks = @import("callbacks.zig");
const Main = @import("main.zig");
const Sliders = @import("sliders.zig");
const Gui = @import("libs/gui_windows.zig");
const Shapes = @import("libs/shapes.zig");
const Consumer = @import("libs/consumer.zig");
const Producer = @import("libs/producer.zig");
const Wgpu = @import("libs/wgpu.zig");

pub const Window = struct {
    pos: Gui.Pos,
    size: Gui.Pos,
    set_window: bool = true,
    p_open: bool = true,
    change: bool = false,
    window_fn: *const fn (demo: *Main) void,
    window_flags: zgui.WindowFlags = .{},
};

fn displayImguiWindow(demo: *Main, window: *Window) void {
    //Make see through
    //zgui.setNextWindowBgAlpha(.{ .alpha = 0 });
    //zgui.pushStyleColor1u(.{ .idx = .frame_bg, .c = 0 });
    //zgui.pushStyleColor1u(.{ .idx = .window_bg, .c = 0 });
    //defer zgui.popStyleColor(.{});
    //defer zgui.popStyleColor(.{});
    if (window.p_open and window.change) {
        window.change = false;
        const sd = demo.gctx.swapchain_descriptor;
        Gui.setupWindowPos(sd, window.pos);
        Gui.setupWindowSize(sd, window.size);
    }
    if (window.p_open) {
        var buf: [100]u8 = undefined;
        const window_id = std.fmt.bufPrintZ(buf[0..], "##{any}", .{window.window_fn}) catch unreachable;
        if (zgui.begin(window_id, .{
            .flags = window.window_flags,
            .popen = &window.p_open,
        })) {
            window.window_fn(demo);
        }
        zgui.end();
    }
}

pub fn update(demo: *Main) void {
    Wgpu.checkObjBufState(u32, &demo.stats.obj_buf.mapping);
    Wgpu.checkObjBufState(Producer, &demo.buffers.producers.mapping);
    Wgpu.checkObjBufState(Consumer, &demo.buffers.consumers.mapping);

    //zgui.showDemoWindow(null);
    displayImguiWindow(demo, &demo.imgui_windows.sliders);
    displayImguiWindow(demo, &demo.imgui_windows.statistics);
    displayImguiWindow(demo, &demo.imgui_windows.help);

    if (demo.running) {
        demo.gctx.queue.writeBuffer(
            demo.gctx.lookupResource(demo.stats.obj_buf.buf).?,
            6 * @sizeOf(u32),
            f32,
            &.{ random.float(f32), random.float(f32) },
        );
        Wgpu.getAllAsync(Producer, Callbacks.price, .{
            .gctx = demo.gctx,
            .obj_buf = &demo.buffers.producers,
            .stat_array = &demo.stats.price,
            .gui_slider = &demo.params.price.val,
        });
        Wgpu.getAllAsync(u32, Callbacks.getNumAgents, .{
            .gctx = demo.gctx,
            .obj_buf = &demo.stats.obj_buf,
            .consumer_num_structs = &demo.buffers.consumers.mapping.num_structs,
            .producer_num_structs = &demo.buffers.producers.mapping.num_structs,
            .gui_slider = &demo.params.num_producers.val,
        });
        const current_time = @as(f32, @floatCast(demo.gctx.stats.time));
        const seconds_passed = current_time - demo.stats.second;
        if (seconds_passed >= 1) {
            demo.stats.second = current_time;
            //Wgpu.getAllAsync(Producer, Callbacks.price, .{
            //    .gctx = demo.gctx,
            //    .obj_buf = &demo.buffers.producers,
            //    .stat_array = &demo.stats.price,
            //    .gui_slider = &demo.params.price.val,
            //});
            Wgpu.getAllAsync(u32, Callbacks.numTransactions, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.stats.obj_buf,
                .stat_array = &demo.stats.num_transactions,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.emptyConsumers, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.consumers,
                .stat_array = &demo.stats.num_empty_consumers,
            });
            Wgpu.getAllAsync(Producer, Callbacks.avgProducerInventory, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.producers,
                .stat_array = &demo.stats.avg_producer_inventory,
            });
            Wgpu.getAllAsync(Producer, Callbacks.avgProducerMoney, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.producers,
                .stat_array = &demo.stats.avg_producer_money,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.avgConsumerInventory, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.consumers,
                .stat_array = &demo.stats.avg_consumer_inventory,
            });
            Wgpu.getAllAsync(Consumer, Callbacks.avgConsumerMoney, .{
                .gctx = demo.gctx,
                .obj_buf = &demo.buffers.consumers,
                .stat_array = &demo.stats.avg_consumer_money,
            });
        }
    }
}

pub fn plots(demo: *Main) void {
    if (zgui.button("Hide Statistics", .{})) {
        demo.imgui_windows.statistics.p_open = !demo.imgui_windows.statistics.p_open;
        if (demo.imgui_windows.statistics.p_open) {
            demo.imgui_windows.statistics.change = true;
        }
    }

    if (zgui.beginTable("Stats", .{ .column = 2 })) {
        defer zgui.endTable();

        zgui.tableSetupColumn("Value", .{ .flags = .{ .width_fixed = true }, .init_width_or_height = 60 });
        zgui.tableSetupColumn("Plot", .{ .flags = .{} });
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
    zgui.text("{any}", .{arr.getLastOrNull()});
    _ = zgui.tableSetColumnIndex(1);
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
    slider: *Sliders.Slider(T),
) bool {
    zgui.textWrapped(name, .{});
    if (!std.mem.eql(u8, "", slider.help)) {
        zgui.sameLine(.{});
        zgui.textDisabled("(?)", .{});
        if (zgui.isItemHovered(.{})) {
            _ = zgui.beginTooltip();
            zgui.textUnformatted(slider.help);
            zgui.endTooltip();
        }
    }
    var buf: [1000]u8 = undefined;
    const id = std.fmt.bufPrintZ(buf[0..], "##{any}{any}", .{ slider.help, name }) catch unreachable;
    return zgui.sliderScalar(
        id,
        T,
        .{
            .v = &slider.val,
            .min = slider.min,
            .max = slider.max,
        },
    );
}

pub fn settings(demo: *Main) void {
    zgui.pushItemWidth(zgui.getContentRegionAvail()[0]);
    zgui.text("  {d:.1} fps", .{demo.gctx.stats.fps});

    const producers = demo.buffers.producers;
    zgui.text("Producer Settings", .{});
    const np = &demo.params.num_producers;
    if (createSlider("Number of Producers", u32, np)) {
        const resource = demo.gctx.lookupResource(demo.stats.obj_buf.buf).?;
        demo.gctx.queue.writeBuffer(
            resource,
            2 * @sizeOf(u32),
            u32,
            &.{np.val},
        );

        if (np.old.? >= np.val) {
            demo.buffers.producers.mapping.num_structs = np.val;
        } else {
            Producer.generateBulk(
                demo.gctx,
                &demo.buffers.producers,
                np.val - np.old.?,
                .{
                    .max_inventory = demo.params.max_inventory.val,
                    .production_cost = demo.params.production_cost.val,
                    .max_production_rate = demo.params.max_production_rate.val,
                    .price = demo.params.price.val,
                    .max_money = demo.params.max_producer_money.val,
                    .decay_rate = demo.params.decay_rate.val,
                },
            );
        }
        np.old.? = np.val;
    }

    const pc = &demo.params.production_cost;
    if (createSlider("Production Cost", u32, pc)) {
        producers.updateU32Field(demo.gctx, pc.val, "production_cost");
    }

    const price = &demo.params.price;
    if (createSlider("Price Sold", u32, price)) {
        producers.updateU32Field(demo.gctx, price.val, "price");
    }

    const mpr = &demo.params.max_production_rate;
    if (createSlider("Max Production Rate", u32, mpr)) {
        producers.updateU32Field(demo.gctx, mpr.val, "max_production_rate");
    }

    const mpm = &demo.params.max_producer_money;
    if (createSlider("Max Money", u32, mpm)) {
        producers.updateU32Field(demo.gctx, mpm.val, "max_money");
        Wgpu.getAllAsync(Producer, Callbacks.updateProducerMoney, .{
            .gctx = demo.gctx,
            .obj_buf = &demo.buffers.producers,
        });
    }

    const mpi = &demo.params.max_inventory;
    if (createSlider("Max Inventory", u32, mpi)) {
        producers.updateU32Field(demo.gctx, mpi.val, "max_inventory");
        Wgpu.getAllAsync(Producer, Callbacks.updateProducerInventory, .{
            .gctx = demo.gctx,
            .obj_buf = &demo.buffers.producers,
        });
    }

    const dr = &demo.params.decay_rate;
    if (createSlider("Decay Rate", u32, dr)) {
        producers.updateU32Field(demo.gctx, dr.val, "decay_rate");
    }

    zgui.dummy(.{ .w = 1.0, .h = 40.0 });

    zgui.text("Consumer Settings", .{});
    const nc = &demo.params.num_consumers;
    if (createSlider("Number of Consumers", u32, nc)) {
        const resource = demo.gctx.lookupResource(demo.stats.obj_buf.buf).?;
        demo.gctx.queue.writeBuffer(
            resource,
            @sizeOf(u32),
            u32,
            &.{nc.val},
        );

        if (nc.old.? >= nc.val) {
            demo.buffers.consumers.mapping.num_structs = nc.val;
        } else {
            Consumer.generateBulk(
                demo.gctx,
                &demo.buffers.consumers,
                nc.val - nc.old.?,
                .{
                    .income = demo.params.income.val,
                    .moving_rate = demo.params.moving_rate.val,
                    .max_money = demo.params.max_consumer_money.val,
                },
            );
        }
        nc.old.? = nc.val;
    }

    const consumers = demo.buffers.consumers;
    const ci = &demo.params.income;
    if (createSlider("Income", u32, &demo.params.income)) {
        consumers.updateU32Field(demo.gctx, ci.val, "income");
    }

    const mcm = &demo.params.max_consumer_money;
    if (createSlider("Max Money", u32, mcm)) {
        consumers.updateU32Field(demo.gctx, mcm.val, "max_money");
        Wgpu.getAllAsync(Consumer, Callbacks.updateConsumerMoney, .{
            .gctx = demo.gctx,
            .obj_buf = &demo.buffers.consumers,
        });
    }

    const mr = &demo.params.moving_rate;
    if (createSlider("Moving Rate", f32, mr)) {
        consumers.updateF32Field(demo.gctx, mr.val, "moving_rate");
    }
    const cs = &demo.params.consumer_size;
    if (createSlider("Consumer Size", f32, cs)) {
        demo.graphics_objects.consumers.vertex_buffer = Shapes.createCircleVertexBuffer(
            demo.gctx,
            Main.NUM_CONSUMER_SIDES,
            cs.val,
        );
        demo.graphics_objects.consumers_money.vertex_buffer = Shapes.createCircleVertexBuffer(
            demo.gctx,
            Main.NUM_CONSUMER_SIDES,
            cs.val,
        );
    }
    const ps = &demo.params.producer_size;
    if (createSlider("Producer Size", f32, ps)) {
        demo.graphics_objects.producers.vertex_buffer = Shapes.createRectangleVertexBuffer(
            demo.gctx,
            ps.val,
            ps.val,
        );
        demo.graphics_objects.producers_bar.vertex_buffer = Shapes.createRectangleVertexBuffer(
            demo.gctx,
            ps.val / 2,
            ps.val * 6,
        );
    }
    zgui.dummy(.{ .w = 1.0, .h = 20.0 });

    if (zgui.button("Start", .{})) {
        demo.running = true;
    }
    zgui.sameLine(.{});
    if (zgui.button("Stop", .{})) {
        demo.running = false;
    }
    if (zgui.button("Restart", .{})) {
        demo.running = true;
        Main.restartSimulation(demo);
    }
    if (zgui.button("Supply Shock", .{})) {
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
        demo.imgui_windows.statistics.p_open = !demo.imgui_windows.statistics.p_open;
        if (demo.imgui_windows.statistics.p_open) {
            demo.imgui_windows.statistics.change = true;
        }
    }

    if (zgui.button("Help Page", .{})) {
        demo.imgui_windows.help.p_open = !demo.imgui_windows.help.p_open;
        if (demo.imgui_windows.help.p_open) {
            demo.imgui_windows.help.change = true;
        }
    }
}

pub fn help(demo: *Main) void {
    if (zgui.button("Close Page", .{})) {
        demo.imgui_windows.help.p_open = !demo.imgui_windows.help.p_open;
        if (demo.imgui_windows.help.p_open) {
            demo.imgui_windows.help.change = true;
        }
    }
    zgui.bulletText("Welcome to Economic Visuals!", .{});
    zgui.text("", .{});
    zgui.bulletText("This simulation has two basic agents: Consumers and Producers.", .{});
    zgui.bulletText("Consumers are the circles. Producers are the squares.", .{});
    zgui.bulletText("Producers create resources and Consumers consume these resources.", .{});
    zgui.bulletText("The size of both Producers and Consumers grows to show how many resources they currently have.", .{});
    zgui.bulletText("This is called their inventory.", .{});
    zgui.bulletText("When Consumers have no inventory they turn red, otherwise they are green.", .{});
    zgui.bulletText("Whenever Consumers are empty they travel to a Producer and try to buy more resources before returning home.", .{});
    zgui.bulletText("Consumers choose the Producer which has the largest inventory from which they can buy.", .{});
    zgui.bulletText("If two Producers have the same inventory then the closest Producer is chosen.", .{});
    zgui.text("", .{});
    zgui.bulletText("Consumers and Producers both have money in this simulation.", .{});
    zgui.bulletText("Consumers have a constant income.", .{});
    zgui.bulletText("Producers only receive money when a consumer buys from them.", .{});
    zgui.bulletText("The price at which this transaction occurs is controlled via the Price Sold slider.", .{});
    zgui.bulletText("Producers use their money to produce resources at the current Production Cost.", .{});
    zgui.bulletText("To keep things constrained there is a maximum amount of money Consumers and Producers can hold.", .{});
    zgui.text("", .{});
    zgui.bulletText("The grey circle around a consumer shows how much it could buy right now at the current price.", .{});
    zgui.bulletText("The white square around a producer shows how much it could produce right now at the current production cost.", .{});
    zgui.text("", .{});
    zgui.bulletText("This simulation is still rather basic.", .{});
    zgui.bulletText("If you have any suggestions, I'd love to hear them!", .{});
}
