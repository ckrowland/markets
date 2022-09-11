# Simulations
GPU Accelerated agent-based modeling to visualize and simulate complex systems.

## Current Simulations
![demo](gh_demo.gif)

- Resource Distribution
  - Simulate market dynamics of consumers and producers.
  - Consumers move to Producers, get resources, travel home and consume the resources.
  - Red means consumer is empty and looking for resources.
  - Green means consumer has enough resources to consume.
  - Parameters:
    - Number of Consumers
    - Number of Producers
    - Consumers moving rate
    - Consumers consumption rate
    - Producers production rate
    - Producers giving rate
    - Producers maximum inventory
  - Data Gathered:
    - Transactions per second
    - Empty Consumers
    - Total Producer Inventory
  - To remove a line from the graph, click it's title in the legend.



## Install
- `git` with [Git LFS](https://git-lfs.github.com/)
- [Zig 0.10.0-dev.3952 (master)](https://ziglang.org/download/) or newer

## Run
```
git clone --recurse-submodules https://github.com/ckrowland/simulations.git
cd simulations
zig build resources-run
```
