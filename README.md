# Markets

Visually simulate markets of basic consumers and producers.
Built on [zig-gamedev](https://github.com/zig-gamedev/zig-gamedev/).
Download from the latest release or build from source.

## Random Resource Simulation

[demo.webm](https://user-images.githubusercontent.com/95145274/202062756-61222967-26ee-41e1-ba2b-fb9d7d2d41a1.webm)

- This simulation has two basic agents: Consumers and Producers.
- Consumers are the circles. Producers are the squares.
- Producers create resources and Consumers consume these resources.
- The size of both Producers and Consumers grows to show how many resources they currently have.
- This is called their inventory.
- When Consumers have no inventory they turn red, otherwise they are green.
- Whenever Consumers are empty they travel to a Producer and try to buy more resources before returning home.
- Consumers choose the Producer which has the largest inventory from which they can buy.
- If two Producers have the same inventory then the closest Producer is chosen.
- Consumers and Producers both have money in this simulation.
- Consumers have a constant income.
- Producers only receive money when a consumer buys from them.
- The price at which this transaction occurs is controlled via the Price Sold slider.
- Producers use their money to produce resources at the current Production Cost.
- To keep things constrained there is a maximum amount of money Consumers and Producers can hold.
- The grey circle around a consumer shows how much it could buy right now at the current price.
- The white square around a producer shows how much it could produce right now at the current production cost.

Parameters:

- Number of Producers
- Production Cost
- Price Sold
- Max Production Rate
- Producer Max Money
- Producer Max Inventory
- Decay Rate
- Number of Consumers
- Consumer income
- Consumer Max Money
- Moving Rate

Data Gathered:

- Transactions per second
- Empty Consumers
- Avg Producer Inventory
- Avg Producer Money
- Avg Consumer Inventory
- Avg Consumer Money
- Total Producer Inventory

## Build From Source

### Download

- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.github.com/)
- Zig **0.14.1**.

### Run

```
git clone https://github.com/ckrowland/markets.git
cd simulations
zig build run
```

## Resource Editor (Currently Broken)

[editor.webm](https://github.com/ckrowland/simulations/assets/95145274/2c21762f-0dd2-4a00-8d2e-0aad38e83c78)

- Manually place position of consumers and producers.
- Each producer and consumer grouping has individual parameters.

## Variable Parameters (Currently Broken)

[variable.webm](https://github.com/ckrowland/simulations/assets/95145274/b7e97f85-6828-42fe-827d-af6ee2bdb049)

- Very similiar to the random simulation.
- Have input parameters controlled via a wave timeline.
