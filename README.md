# Simulations
GPU accelerated visual simulations.
Built on [zig-gamedev](https://github.com/michal-z/zig-gamedev/).

## Table of Contents
* Current Simulations
  * Random Resource Simulation - Generate economy via input parameters
  * Resource Editor - Manually create economy with individualized parameters
* Build - Build and run a native application 

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
  - Total Producer Inventory
- To remove a line from the graph, click it's title in the legend.

## Resource Editor
[editor.webm](https://github.com/ckrowland/simulations/assets/95145274/2c21762f-0dd2-4a00-8d2e-0aad38e83c78)

- Manually place position of consumers and producers.
- Each producer and consumer grouping has individual parameters.


## Build

### Download
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.github.com/)
- Zig **0.12.0-dev.1871+e426ae43a** 

| OS/Arch         | Download link               |
| --------------- | --------------------------- |
| Windows x86_64  | [zig-windows-x86_64-0.12.0-dev.1871+e426ae43a.zip](https://ziglang.org/builds/zig-windows-x86_64-0.12.0-dev.1871+e426ae43a.zip) |
| Linux x86_64    | [zig-linux-x86_64-0.12.0-dev.1871+e426ae43a.tar.xz](https://ziglang.org/builds/zig-linux-x86_64-0.12.0-dev.1871+e426ae43a.tar.xz) |
| macOS x86_64    | [zig-macos-x86_64-0.12.0-dev.1871+e426ae43a.tar.xz](https://ziglang.org/builds/zig-macos-x86_64-0.12.0-dev.1871+e426ae43a.tar.xz) |
| macOS aarch64   | [zig-macos-aarch64-0.12.0-dev.1871+e426ae43a.tar.xz](https://ziglang.org/builds/zig-macos-aarch64-0.12.0-dev.1871+e426ae43a.tar.xz) |

### Run
```
git clone https://github.com/ckrowland/simulations.git
cd simulations
zig build random-run # Runs Natively
zig build editor-run # Runs Natively
zig build random-web # Generates web files
zig build editor-web # Generates web files
```
