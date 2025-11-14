# Markets

Visually simulate markets of basic consumers and producers.
Built on [zig-gamedev](https://github.com/zig-gamedev/zig-gamedev/).
Download from the latest release or build from source.

## Random Resource Simulation

[demo.webm](https://user-images.githubusercontent.com/95145274/202062756-61222967-26ee-41e1-ba2b-fb9d7d2d41a1.webm)

- Simulate market dynamics of consumers and producers.
- Consumers move to Producers, get resources, travel home and consume the resources.
- Red means consumer is empty and looking for resources.
- Green means consumer has enough resources to consume.
- Parameters:
  - Number of Consumers
  - Number of Producers
  - Consumers moving rate
  - Consumers demand rate
  - Consumers size
  - Producers production rate
  - Producers maximum inventory
- Data Gathered:
  - Transactions per second
  - Empty Consumers
  - Avg Producer Inventory
  - Avg Producer Money
  - Avg Consumer Inventory
  - Avg Consumer Money
  - Total Producer Inventory
- To remove a line from the graph, click it's title in the legend.

## Build From Source

### Download

- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.github.com/)
- Zig **0.14.1**.

### Run

```
git clone https://github.com/ckrowland/simulations.git
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
