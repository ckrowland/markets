# Simulations
GPU accelerated visual simulations.

### Resource Distribution

![demo](gh_demo.gif)

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
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.github.com/)
- [Zig (master)](https://ziglang.org/download/)

## Run
```
git clone https://github.com/ckrowland/simulations.git
cd simulations
git submodule update --init --remote
zig build resources-run
```

## Libraries Used
- [zig-gamedev](https://github.com/michal-z/zig-gamedev/)

