# MCP System Monitor Server

A cross-platform MCP (Model Context Protocol) server that provides comprehensive real-time system monitoring
capabilities for LLMs. Built with FastMCP for easy integration with Claude Desktop and other MCP-compatible clients.

## Features

### System Monitoring

**Basic System Monitoring:**
- **CPU Monitoring**: Real-time usage, per-core statistics, frequency, temperature, detailed processor information (model, vendor, architecture, cache sizes)
- **GPU Monitoring**: Multi-vendor GPU support (NVIDIA with full metrics, Apple Silicon with comprehensive support including unified memory and core count, AMD/Intel with basic info)
- **Memory Monitoring**: RAM and swap usage, availability statistics
- **Disk Monitoring**: Space usage, filesystem information for all mounted drives
- **Network Statistics**: Interface-level traffic and error counters
- **Process Monitoring**: Top processes by CPU/memory usage
- **System Information**: OS details, hostname, uptime, architecture

**Phase 1 Performance Monitoring:**
- **I/O Performance**: Detailed disk I/O metrics, read/write rates, per-disk statistics, busy time analysis
- **System Load**: Load averages (1m, 5m, 15m), context switches, interrupts, running/blocked processes
- **Enhanced Memory**: Detailed memory statistics including buffers, cache, active/inactive memory, page faults, swap activity
- **Enhanced Network**: Network performance metrics with transfer rates, errors, drops, interface speed and MTU

### MCP Tools Available

**Basic System Monitoring (9 tools):**
- `get_current_datetime`: Get the current local datetime in ISO format
- `get_cpu_info`: Get current CPU usage and statistics
- `get_gpu_info`: Get GPU information for all detected GPUs
- `get_memory_info`: Get RAM and swap usage
- `get_disk_info`: Get disk usage for all mounted drives
- `get_system_snapshot`: Get complete system state in one call
- `monitor_cpu_usage`: Monitor CPU usage over a specified duration
- `get_top_processes`: Get top processes by CPU or memory usage
- `get_network_stats`: Get network interface statistics

**Phase 1 Performance Monitoring (6 tools):**
- `get_io_performance`: Get detailed I/O performance metrics and rates
- `get_system_load`: Get system load averages and process statistics
- `get_enhanced_memory_info`: Get detailed memory statistics with caches/buffers
- `get_enhanced_network_stats`: Get enhanced network performance metrics
- `get_performance_snapshot`: Get complete performance monitoring snapshot
- `monitor_io_performance`: Monitor I/O performance over specified duration with trend analysis

### MCP Resources

**Basic System Resources (3 resources):**
- `system://live/cpu`: Live CPU usage data
- `system://live/memory`: Live memory usage data
- `system://config`: System configuration and hardware information

**Phase 1 Performance Resources (3 resources):**
- `system://performance/io`: Live I/O performance data
- `system://performance/load`: Live system load data
- `system://performance/network`: Live network performance data

### GPU Support Details

**NVIDIA GPUs:**
- Full metrics: usage percentage, memory (used/total), temperature, power consumption
- Supports multiple NVIDIA GPUs
- Requires NVIDIA drivers and NVML libraries

**Apple Silicon GPUs:**
- Comprehensive support for M1, M2, and M3 chips
- GPU core count detection
- Unified memory reporting (shares system RAM)
- Metal API support detection
- Temperature monitoring (when available)

**AMD/Intel GPUs:**
- Basic detection and identification
- Limited metrics depending on platform and drivers

## Requirements

- Python 3.10+
- Windows, macOS, or Linux
- GPU (optional): NVIDIA GPUs for full metrics, Apple Silicon GPUs fully supported on macOS

## Installation

### From GitHub

1. Clone the repository:
   ```bash
   git clone https://github.com/huhabla/mcp-system-monitor.git
   cd mcp-system-monitor
   ```

2. Install dependencies using uv (recommended):
   ```bash
   uv pip install -e .
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

### Optional Dependencies

For Windows-specific features:

```bash
pip install mcp-system-monitor[win32]
```

## Usage

### Development Mode

Test the server with the MCP Inspector:

```bash
uv run mcp dev mcp_system_monitor_server.py
```

### Claude Desktop Integration

Install the server in Claude Desktop:

```bash
uv run mcp install mcp_system_monitor_server.py --name "System Monitor"
```

### Direct Execution

Run the server directly:

```bash
python mcp_system_monitor_server.py
```

### MCP Servers Json Config

Modify the following JSON template to set the path to the MCP server in your MCP client for Windows:

```json
{
  "mcpServers": {
    "mpc-system-monitor": {
      "command": "cmd",
      "args": [
        "/c",
        "C:/Users/Sören Gebbert/Documents/GitHub/mcp-system-monitor/start_mpc_system_monitor.bat"
      ]
    }
  }
}
```

Modify the following JSON template to set the path to the MCP server in your MCP client for MacOS:

```json
{
  "mcpServers": {
    "mpc-system-monitor": {
      "command": "/bin/zsh",
      "args": [
        "/Users/holistech/Documents/GitHub/mcp-system-monitor/start_mcp_system_monitor.sh"
      ]
    }
  }
}
```

### Example Tool Usage

Once connected to Claude Desktop or another MCP client, you can use natural language to interact with the system
monitor:

**Basic System Monitoring:**
- "Show me the current CPU usage"
- "What's my GPU temperature?"
- "How many GPU cores does my Apple M1 Max have?"
- "Show me GPU memory usage and whether it's unified memory"
- "How much disk space is available?"
- "Monitor CPU usage for the next 10 seconds"
- "Show me the top 5 processes by memory usage"
- "Get a complete system snapshot"

**Phase 1 Performance Monitoring:**
- "Show me detailed I/O performance metrics"
- "What's the current system load average?"
- "Monitor I/O performance for the next 30 seconds"
- "Show me enhanced memory statistics with cache information"
- "Get detailed network performance metrics"
- "Give me a complete performance snapshot"

## Architecture

The server uses a modular collector-based architecture:

- **BaseCollector**: Abstract base class providing caching and async data collection
- **Specialized Collectors**: CPU, GPU, Memory, Disk, Network, Process, and System collectors
- **Phase 1 Performance Collectors**: IOPerformance, SystemLoad, EnhancedMemory, and EnhancedNetwork collectors
- **Pydantic Models**: Type-safe data models for all system information
- **FastMCP Integration**: Simple decorators for exposing tools and resources

### Caching Strategy

All collectors implement intelligent caching to:

- Reduce system overhead from frequent polling
- Provide consistent data within time windows
- Allow configurable cache expiration

## Testing

### Comprehensive Test Suite

The project includes a comprehensive test suite with 100% coverage of all MCP tools, resources, and collectors:

**Test Organization:**
- **`test_mcp_system_monitor_server.py`** - Original basic collector tests
- **`test_mcp_system_monitor_server_comprehensive.py`** - Comprehensive MCP tools/resources tests
- **`test_mcp_server_integration.py`** - Integration tests for MCP server protocol compliance
- **`test_architecture_agnostic.py`** - Cross-platform tests focusing on data contracts
- **`conftest.py`** - Test configuration, fixtures, and mocking utilities

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run tests by category:**
```bash
pytest -m unit              # Fast unit tests only
pytest -m integration       # Integration tests only
pytest -m agnostic          # Architecture/OS agnostic tests
pytest -m "not slow"        # Exclude slow tests
pytest -m "unit and not slow"  # Fast unit tests for CI
```

**Run specific test suites:**
```bash
pytest tests/test_mcp_system_monitor_server_comprehensive.py  # All MCP endpoints
pytest tests/test_mcp_server_integration.py                  # Integration tests
pytest tests/test_architecture_agnostic.py                   # Cross-platform tests
```

**Run with coverage:**
```bash
pytest --cov=mcp_system_monitor_server --cov-report=html
```

### Test Coverage

**Complete Coverage:**
- **15 MCP Tools** (9 basic + 6 Phase 1 performance)
- **6 MCP Resources** (3 basic + 3 Phase 1 performance)
- **11 Collectors** (7 basic + 4 Phase 1 performance)
- **Cross-platform compatibility** testing
- **Performance benchmarking** and stress testing
- **Error handling** and edge case validation

**Performance Benchmarks:**
- System snapshot collection: < 5 seconds
- Individual tool calls: < 1 second each
- Concurrent operations: 20 parallel calls < 10 seconds

## Platform Support

| Feature                 | Windows | macOS | Linux |
|-------------------------|---------|-------|-------|
| CPU Monitoring          | ✅       | ✅     | ✅     |
| GPU Monitoring (NVIDIA) | ✅       | ✅     | ✅     |
| GPU Monitoring (AMD)    | ⚠️      | ❌     | ⚠️    |
| GPU Monitoring (Intel)  | ⚠️      | ❌     | ⚠️    |
| GPU Monitoring (Apple)  | ❌       | ✅     | ❌     |
| Memory Monitoring       | ✅       | ✅     | ✅     |
| Disk Monitoring         | ✅       | ✅     | ✅     |
| Network Statistics      | ✅       | ✅     | ✅     |
| Process Monitoring      | ✅       | ✅     | ✅     |
| CPU Temperature         | ⚠️      | ⚠️    | ✅     |

⚠️ = Limited support, depends on hardware/drivers

## Troubleshooting

### GPU Monitoring Not Working

**NVIDIA GPUs:**
- Ensure NVIDIA drivers are installed
- Check if `nvidia-smi` command works
- The server will gracefully handle missing GPU libraries

**Apple Silicon GPUs:**
- Supported on macOS with M1, M2, and M3 chips
- Provides comprehensive information including unified memory and GPU core count
- Uses `system_profiler` command (available by default on macOS)

### Permission Errors

- Some system information may require elevated privileges
- The server handles permission errors gracefully and skips inaccessible resources

### High CPU Usage

- Adjust the monitoring frequency by modifying collector update intervals
- Use cached data methods to reduce system calls
- Default cache expiration is 2 seconds for most collectors
- Consider increasing `max_age` parameter in `get_cached_data()` calls for less frequent updates

### Performance Considerations

- The server uses intelligent caching to minimize system calls
- Each collector maintains its own cache with configurable expiration
- Continuous monitoring tools (like `monitor_cpu_usage`) bypass caching for real-time data
- For high-frequency polling, consider using the resource endpoints which leverage caching

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastMCP](https://github.com/modelcontextprotocol/python-sdk)
- System monitoring via [psutil](https://github.com/giampaolo/psutil)
- NVIDIA GPU support via [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/)
