# MCP System Monitor Server

A cross-platform MCP (Model Context Protocol) server that provides comprehensive real-time system monitoring
capabilities for LLMs. Built with FastMCP for easy integration with Claude Desktop and other MCP-compatible clients.

## Features

### System Monitoring

- **CPU Monitoring**: Real-time usage, per-core statistics, frequency, temperature, detailed processor information (model, vendor, architecture, cache sizes)
- **GPU Monitoring**: NVIDIA GPU support via NVML (usage, VRAM, temperature)
- **Memory Monitoring**: RAM and swap usage, availability statistics
- **Disk Monitoring**: Space usage, filesystem information for all mounted drives
- **Network Statistics**: Interface-level traffic and error counters
- **Process Monitoring**: Top processes by CPU/memory usage
- **System Information**: OS details, hostname, uptime, architecture

### MCP Tools Available

- `get_cpu_info`: Get current CPU usage and statistics
- `get_gpu_info`: Get GPU information for all detected GPUs
- `get_memory_info`: Get RAM and swap usage
- `get_disk_info`: Get disk usage for all mounted drives
- `get_system_snapshot`: Get complete system state in one call
- `monitor_cpu_usage`: Monitor CPU usage over a specified duration
- `get_top_processes`: Get top processes by CPU or memory usage
- `get_network_stats`: Get network interface statistics

### MCP Resources

- `system://live/cpu`: Live CPU usage data
- `system://live/memory`: Live memory usage data
- `system://config`: System configuration and hardware information

## Requirements

- Python 3.10+
- Windows, macOS, or Linux
- NVIDIA GPU (optional, for GPU monitoring)

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

### MCP Json config

Modify the following JSON template to set the path to the MCP server in your MCP client:

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

### Example Tool Usage

Once connected to Claude Desktop or another MCP client, you can use natural language to interact with the system
monitor:

- "Show me the current CPU usage"
- "What's my GPU temperature?"
- "How much disk space is available?"
- "Monitor CPU usage for the next 10 seconds"
- "Show me the top 5 processes by memory usage"
- "Get a complete system snapshot"

## Architecture

The server uses a modular collector-based architecture:

- **BaseCollector**: Abstract base class providing caching and async data collection
- **Specialized Collectors**: CPU, GPU, Memory, Disk, Network, Process, and System collectors
- **Pydantic Models**: Type-safe data models for all system information
- **FastMCP Integration**: Simple decorators for exposing tools and resources

### Caching Strategy

All collectors implement intelligent caching to:

- Reduce system overhead from frequent polling
- Provide consistent data within time windows
- Allow configurable cache expiration

## Testing

Run the test suite:

```bash
pytest tests/test_mcp_system_monitor_server.py -v
```

Run with coverage:

```bash
pytest tests/test_mcp_system_monitor_server.py --cov=mcp_system_monitor_server --cov-report=html
```

## Platform Support

| Feature                 | Windows | macOS | Linux |
|-------------------------|---------|-------|-------|
| CPU Monitoring          | ✅       | ✅     | ✅     |
| GPU Monitoring (NVIDIA) | ✅       | ✅     | ✅     |
| Memory Monitoring       | ✅       | ✅     | ✅     |
| Disk Monitoring         | ✅       | ✅     | ✅     |
| Network Statistics      | ✅       | ✅     | ✅     |
| Process Monitoring      | ✅       | ✅     | ✅     |
| CPU Temperature         | ⚠️      | ⚠️    | ✅     |

⚠️ = Limited support, depends on hardware/drivers

## Troubleshooting

### GPU Monitoring Not Working

- Ensure NVIDIA drivers are installed
- Check if `nvidia-smi` command works
- The server will gracefully handle missing GPU libraries

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
- 