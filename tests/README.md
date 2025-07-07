# MCP System Monitor Test Suite

This directory contains a comprehensive test suite for the MCP System Monitor Server.

## Test Organization

### Test Files

- **`test_mcp_system_monitor_server.py`** - Original basic collector tests
- **`test_mcp_system_monitor_server_comprehensive.py`** - Comprehensive tests for all MCP tools, resources, and collectors  
- **`test_mcp_server_integration.py`** - Integration tests for MCP server functionality and protocol compliance
- **`test_architecture_agnostic.py`** - Architecture and OS independent tests focusing on data contracts
- **`conftest.py`** - Test configuration, fixtures, and shared test utilities

### Test Categories

Tests are organized using pytest markers:

- **`@pytest.mark.unit`** - Fast, isolated unit tests
- **`@pytest.mark.integration`** - Integration tests that test component interactions
- **`@pytest.mark.slow`** - Tests that may take several seconds to complete
- **`@pytest.mark.agnostic`** - Architecture and OS agnostic tests
- **`@pytest.mark.asyncio`** - Async tests (automatically applied to async functions)

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories
```bash
# Run only fast unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run architecture-agnostic tests
pytest -m agnostic

# Skip slow tests
pytest -m "not slow"
```

### Run Specific Test Files
```bash
# Run comprehensive MCP endpoint tests
pytest tests/test_mcp_system_monitor_server_comprehensive.py

# Run integration tests
pytest tests/test_mcp_server_integration.py

# Run original collector tests
pytest tests/test_mcp_system_monitor_server.py
```

### Coverage Reports
```bash
# Run with coverage
pytest --cov=mcp_system_monitor_server --cov-report=html

# Generate detailed coverage report
pytest --cov=mcp_system_monitor_server --cov-report=term-missing --cov-report=html
```

## Test Coverage

### MCP Tools Tested ✅
- `get_current_datetime` - Date/time functionality
- `get_cpu_info` - CPU information and metrics
- `get_gpu_info` - GPU detection and monitoring  
- `get_memory_info` - Memory usage statistics
- `get_disk_info` - Disk space and filesystem information
- `get_system_snapshot` - Complete system state capture
- `monitor_cpu_usage` - CPU monitoring over time
- `get_top_processes` - Process monitoring and ranking
- `get_network_stats` - Network interface statistics

### MCP Resources Tested ✅
- `system://live/cpu` - Live CPU usage data
- `system://live/memory` - Live memory usage data  
- `system://config` - Static system configuration

### Collectors Tested ✅
- `CPUCollector` - CPU data collection with caching
- `GPUCollector` - Multi-platform GPU detection
- `MemoryCollector` - Memory and swap statistics
- `DiskCollector` - Disk usage across all mounts
- `SystemCollector` - System information and uptime
- `ProcessCollector` - Process enumeration and ranking
- `NetworkCollector` - Network interface statistics

### Cross-Cutting Concerns Tested ✅
- **Error Handling** - Graceful degradation and error recovery
- **Data Validation** - Pydantic model validation and constraints
- **Performance** - Response times and resource usage
- **Concurrency** - Multiple simultaneous requests
- **Platform Support** - Cross-platform compatibility testing
- **Caching** - Collector caching behavior and cache invalidation
- **Integration** - End-to-end MCP server functionality

## Test Data and Mocking

The test suite uses extensive mocking to ensure consistent, reliable tests:

### Fixtures Available
- `mock_psutil_cpu` - Mocked CPU metrics
- `mock_psutil_memory` - Mocked memory statistics  
- `mock_psutil_disk` - Mocked disk usage data
- `mock_psutil_network` - Mocked network interface data
- `mock_psutil_processes` - Mocked process list
- `mock_platform_info` - Mocked system platform information
- `mock_datetime` - Fixed datetime for consistent testing

### Test Data Characteristics
- **Realistic Values** - Test data reflects real-world system metrics
- **Edge Cases** - Tests include boundary conditions and error scenarios
- **Cross-Platform** - Mocks handle Windows, macOS, and Linux differences
- **Performance** - Tests validate reasonable response times and resource usage

## Performance Testing

### Benchmarks
- System snapshot collection: < 5 seconds
- Individual tool calls: < 1 second each
- Concurrent operations: 20 parallel calls < 10 seconds
- Monitoring overhead: Minimal impact on system resources

### Stress Testing  
- Rapid consecutive calls (10+ per second)
- Mixed workload simulation
- Long-running monitoring sessions
- Large dataset handling (1000+ processes)

## Error Scenarios Tested

### Network and I/O Errors
- Permission denied accessing system files
- Missing or inaccessible hardware components
- Network interface enumeration failures

### Data Validation Errors
- Invalid input parameters to tools
- Malformed system data
- Missing dependencies (NVIDIA libraries, etc.)

### Concurrency Issues
- Race conditions in data collection
- Cache consistency under load
- Resource cleanup and lifecycle management

## Continuous Integration

The test suite is designed for CI/CD environments:

### Fast Feedback
```bash
# Quick smoke test (< 30 seconds)
pytest -m "unit and not slow"
```

### Full Validation  
```bash
# Complete test suite (2-5 minutes)
pytest -m "not slow" --cov=mcp_system_monitor_server
```

### Thorough Testing
```bash  
# All tests including performance (5-10 minutes)
pytest --cov=mcp_system_monitor_server --cov-report=html
```

## Adding New Tests

### For New MCP Tools
1. Add tool test to `test_mcp_system_monitor_server_comprehensive.py`
2. Add integration test to `test_mcp_server_integration.py`  
3. Include performance and error handling scenarios
4. Update this README with new test coverage

### For New Collectors
1. Add collector test to appropriate test file
2. Include comprehensive data validation
3. Test caching behavior and error handling
4. Add realistic mock data fixture if needed

### Test Quality Guidelines
- **Comprehensive** - Test happy path, edge cases, and error conditions
- **Fast** - Unit tests should complete in milliseconds
- **Reliable** - Tests should not be flaky or dependent on external factors
- **Maintainable** - Use fixtures and helpers to reduce duplication
- **Documented** - Include clear docstrings explaining test purpose