# Markets
Visually simulate markets of basic consumers and producers.
Built on [zig-gamedev](https://github.com/michal-z/zig-gamedev/).
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
  - Total Producer Inventory
- To remove a line from the graph, click it's title in the legend.

## Resource Editor
[editor.webm](https://github.com/ckrowland/simulations/assets/95145274/2c21762f-0dd2-4a00-8d2e-0aad38e83c78)
- Manually place position of consumers and producers.
- Each producer and consumer grouping has individual parameters.

## Variable Parameters
[variable.webm](https://github.com/ckrowland/simulations/assets/95145274/b7e97f85-6828-42fe-827d-af6ee2bdb049)
- Very similiar to the random simulation.
- Have input parameters controlled via a wave timeline.

## Build From Source

### Download
- [Git](https://git-scm.com/)
- [Git LFS](https://git-lfs.github.com/)
- Zig **0.13.0-dev.351+64ef45eb0**.

[zigup](https://github.com/marler8997/zigup) is recommended for managing compiler versions. Alternatively, you can download and install manually using the links below:

| OS/Arch         | Download link               |
| --------------- | --------------------------- |
| Windows x86_64  | [zig-windows-x86_64-0.13.0-dev.351+64ef45eb0.zip](https://ziglang.org/builds/zig-windows-x86_64-0.13.0-dev.351+64ef45eb0.zip) |
| Linux x86_64    | [zig-linux-x86_64-0.13.0-dev.351+64ef45eb0.tar.xz](https://ziglang.org/builds/zig-linux-x86_64-0.13.0-dev.351+64ef45eb0.tar.xz) |
| macOS x86_64    | [zig-macos-x86_64-0.13.0-dev.351+64ef45eb0.tar.xz](https://ziglang.org/builds/zig-macos-x86_64-0.13.0-dev.351+64ef45eb0.tar.xz) |
| macOS aarch64   | [zig-macos-aarch64-0.13.0-dev.351+64ef45eb0.tar.xz](https://ziglang.org/builds/zig-macos-aarch64-0.13.0-dev.351+64ef45eb0.tar.xz) |

### Run
```
git clone https://github.com/ckrowland/simulations.git
cd simulations
# Run natively
zig build random-run
zig build editor-run
zig build variable-run

# Run in browser
zig build -Dtarget=wasm32-emscripten random-web-emrun
zig build -Dtarget=wasm32-emscripten editor-web-emrun
zig build -Dtarget=wasm32-emscripten variable-web-emrun
```
