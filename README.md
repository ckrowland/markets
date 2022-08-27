# Simulations
Use GPU Accelerated agent-based modeling to visualize and simulate complex systems.

## Current Simulations
- Resource Distribution
  - Simulate market dynamics of consumers and producers with certain parameters
  - Consumers move to Producers, get resources, travel home and consume the resources.
  - Parameters:
    - Number of Consumers
    - Number of Producers
    - Consumers moving rate
    - Consumers consumption rate
    - Producers production rate
    - Producers giving rate
## Install
- `git` with [Git LFS](https://git-lfs.github.com/)
- [Zig 0.10.0-dev.3027 (master)](https://ziglang.org/download/) or newer

## Run
```
git clone --recurse-submodules https://github.com/ckrowland/simulations.git
cd simulations
zig build resources-run
```
