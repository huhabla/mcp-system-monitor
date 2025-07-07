"""
Architecture and OS agnostic tests for MCP System Monitor.
These tests validate data contracts and behavior without making assumptions
about specific hardware, OS, or architecture characteristics.
"""
import pytest
import asyncio
from datetime import datetime
from typing import Any, Dict, List
from mcp_system_monitor_server import (
    # MCP Tools
    get_current_datetime, get_cpu_info, get_gpu_info, get_memory_info,
    get_disk_info, get_system_snapshot, monitor_cpu_usage, 
    get_top_processes, get_network_stats,
    # Phase 1 MCP Tools
    get_io_performance, get_system_load, get_enhanced_memory_info,
    get_enhanced_network_stats, get_performance_snapshot, monitor_io_performance,
    # MCP Resources  
    live_cpu_resource, live_memory_resource, system_config_resource,
    # Phase 1 MCP Resources
    live_io_performance_resource, live_system_load_resource, live_network_performance_resource,
    # Data Models
    CPUInfo, GPUInfo, MemoryInfo, DiskInfo, SystemInfo, SystemSnapshot,
    # Phase 1 Data Models
    IOPerformanceInfo, SystemLoadInfo, EnhancedMemoryInfo, EnhancedNetworkInfo, SystemPerformanceSnapshot
)


# =============================================================================
# GENERIC DATA CONTRACT TESTS
# =============================================================================

pytestmark = pytest.mark.agnostic

@pytest.mark.asyncio
async def test_get_current_datetime_contract():
    """Test datetime tool returns valid ISO format string"""
    result = await get_current_datetime()
    
    # Must be a string
    assert isinstance(result, str)
    # Must be non-empty
    assert len(result) > 0
    # Must be parseable as datetime
    parsed = datetime.fromisoformat(result.replace(' ', 'T'))
    assert isinstance(parsed, datetime)


@pytest.mark.asyncio
async def test_get_cpu_info_contract():
    """Test CPU info returns valid CPUInfo model"""
    result = await get_cpu_info()
    
    # Must be CPUInfo instance
    assert isinstance(result, CPUInfo)
    
    # Required fields must exist and be reasonable
    assert isinstance(result.usage_percent, (int, float))
    assert isinstance(result.usage_per_core, list)
    assert isinstance(result.core_count, int) and result.core_count > 0
    assert isinstance(result.thread_count, int) and result.thread_count > 0
    assert isinstance(result.processor_name, str) and len(result.processor_name) > 0
    assert isinstance(result.vendor, str) and len(result.vendor) > 0
    assert isinstance(result.architecture, str) and len(result.architecture) > 0
    assert isinstance(result.bits, int) and result.bits > 0
    
    # Logical relationships that should hold universally
    assert result.thread_count >= result.core_count
    assert len(result.usage_per_core) > 0
    assert all(isinstance(usage, (int, float)) for usage in result.usage_per_core)


@pytest.mark.asyncio
async def test_get_gpu_info_contract():
    """Test GPU info returns valid list of GPUInfo models"""
    result = await get_gpu_info()
    
    # Must be a list
    assert isinstance(result, list)
    
    # Each item must be GPUInfo
    for gpu in result:
        assert isinstance(gpu, GPUInfo)
        assert isinstance(gpu.name, str) and len(gpu.name) > 0
        assert isinstance(gpu.usage_percent, (int, float))
        assert isinstance(gpu.memory_used_mb, int) and gpu.memory_used_mb >= 0
        assert isinstance(gpu.memory_total_mb, int) and gpu.memory_total_mb >= 0
        # Optional fields can be None
        assert gpu.temperature is None or isinstance(gpu.temperature, (int, float))
        assert gpu.power_usage is None or isinstance(gpu.power_usage, (int, float))


@pytest.mark.asyncio
async def test_get_memory_info_contract():
    """Test memory info returns valid MemoryInfo model"""
    result = await get_memory_info()
    
    # Must be MemoryInfo instance
    assert isinstance(result, MemoryInfo)
    
    # Required fields with universal constraints
    assert isinstance(result.total_gb, (int, float)) and result.total_gb > 0
    assert isinstance(result.available_gb, (int, float)) and result.available_gb >= 0
    assert isinstance(result.used_gb, (int, float)) and result.used_gb >= 0
    assert isinstance(result.usage_percent, (int, float))
    assert isinstance(result.swap_total_gb, (int, float)) and result.swap_total_gb >= 0
    assert isinstance(result.swap_used_gb, (int, float)) and result.swap_used_gb >= 0


@pytest.mark.asyncio
async def test_get_disk_info_contract():
    """Test disk info returns valid list of DiskInfo models"""
    result = await get_disk_info()
    
    # Must be a list with at least one disk
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Each disk must be valid
    for disk in result:
        assert isinstance(disk, DiskInfo)
        assert isinstance(disk.mount_point, str) and len(disk.mount_point) > 0
        assert isinstance(disk.total_gb, (int, float)) and disk.total_gb > 0
        assert isinstance(disk.used_gb, (int, float)) and disk.used_gb >= 0
        assert isinstance(disk.free_gb, (int, float)) and disk.free_gb >= 0
        assert isinstance(disk.usage_percent, (int, float))
        assert isinstance(disk.filesystem, str) and len(disk.filesystem) > 0


@pytest.mark.asyncio
async def test_get_system_snapshot_contract():
    """Test system snapshot returns complete valid SystemSnapshot"""
    result = await get_system_snapshot()
    
    # Must be SystemSnapshot instance
    assert isinstance(result, SystemSnapshot)
    
    # Must contain all required components
    assert isinstance(result.system, SystemInfo)
    assert isinstance(result.cpu, CPUInfo)
    assert isinstance(result.memory, MemoryInfo)
    assert isinstance(result.gpus, list)
    assert isinstance(result.disks, list)
    assert isinstance(result.collection_time, datetime)
    
    # System info validation
    assert isinstance(result.system.hostname, str) and len(result.system.hostname) > 0
    assert isinstance(result.system.platform, str) and len(result.system.platform) > 0
    assert isinstance(result.system.architecture, str) and len(result.system.architecture) > 0
    assert isinstance(result.system.uptime_seconds, int) and result.system.uptime_seconds >= 0


@pytest.mark.asyncio
async def test_monitor_cpu_usage_contract():
    """Test CPU monitoring returns valid monitoring data"""
    result = await monitor_cpu_usage(duration_seconds=1)
    
    # Must be a dictionary with specific structure
    assert isinstance(result, dict)
    required_keys = ["average", "minimum", "maximum", "samples"]
    assert all(key in result for key in required_keys)
    
    # Values must be numeric
    assert isinstance(result["average"], (int, float))
    assert isinstance(result["minimum"], (int, float))
    assert isinstance(result["maximum"], (int, float))
    assert isinstance(result["samples"], list)
    
    # Logical relationships
    assert result["minimum"] <= result["average"] <= result["maximum"]
    assert len(result["samples"]) > 0
    assert all(isinstance(sample, (int, float)) for sample in result["samples"])


@pytest.mark.asyncio
async def test_get_top_processes_contract():
    """Test process listing returns valid process data"""
    result = await get_top_processes(limit=5)
    
    # Must be a list
    assert isinstance(result, list)
    assert len(result) <= 5
    
    # Each process must have expected structure
    for process in result:
        assert isinstance(process, dict)
        # Don't assume specific fields exist, just validate what's there
        for key, value in process.items():
            assert isinstance(key, str)
            # Values can be various types, just ensure they're not None for basic fields
            if key in ['pid', 'name']:
                assert value is not None


@pytest.mark.asyncio
async def test_get_network_stats_contract():
    """Test network stats returns valid network data"""
    result = await get_network_stats()
    
    # Must be a dictionary
    assert isinstance(result, dict)
    assert "interfaces" in result
    assert isinstance(result["interfaces"], dict)
    
    # Each interface must have valid structure
    for interface_name, stats in result["interfaces"].items():
        assert isinstance(interface_name, str) and len(interface_name) > 0
        assert isinstance(stats, dict)
        
        # Basic network stats should be integers
        numeric_fields = ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv']
        for field in numeric_fields:
            if field in stats:
                assert isinstance(stats[field], int) and stats[field] >= 0


# =============================================================================
# PHASE 1 PERFORMANCE MONITORING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_get_io_performance_contract():
    """Test I/O performance tool returns valid IOPerformanceInfo model"""
    result = await get_io_performance()
    
    # Must be IOPerformanceInfo instance
    assert isinstance(result, IOPerformanceInfo)
    
    # Required fields must exist and be reasonable
    assert isinstance(result.read_bytes_per_sec, (int, float)) and result.read_bytes_per_sec >= 0
    assert isinstance(result.write_bytes_per_sec, (int, float)) and result.write_bytes_per_sec >= 0
    assert isinstance(result.read_count_per_sec, (int, float)) and result.read_count_per_sec >= 0
    assert isinstance(result.write_count_per_sec, (int, float)) and result.write_count_per_sec >= 0
    assert isinstance(result.read_time_ms, (int, float)) and result.read_time_ms >= 0
    assert isinstance(result.write_time_ms, (int, float)) and result.write_time_ms >= 0
    assert isinstance(result.busy_time_percent, (int, float)) and 0 <= result.busy_time_percent <= 100
    assert isinstance(result.per_disk_stats, list)
    
    # Per-disk stats validation
    for disk_stat in result.per_disk_stats:
        assert isinstance(disk_stat, dict)
        assert 'device' in disk_stat
        assert isinstance(disk_stat['device'], str) and len(disk_stat['device']) > 0


@pytest.mark.asyncio
async def test_get_system_load_contract():
    """Test system load tool returns valid SystemLoadInfo model"""
    result = await get_system_load()
    
    # Must be SystemLoadInfo instance
    assert isinstance(result, SystemLoadInfo)
    
    # Required fields must exist and be reasonable
    if result.load_average_1m is not None:
        assert isinstance(result.load_average_1m, (int, float)) and result.load_average_1m >= 0
    if result.load_average_5m is not None:
        assert isinstance(result.load_average_5m, (int, float)) and result.load_average_5m >= 0
    if result.load_average_15m is not None:
        assert isinstance(result.load_average_15m, (int, float)) and result.load_average_15m >= 0
    assert isinstance(result.context_switches_per_sec, (int, float)) and result.context_switches_per_sec >= 0
    assert isinstance(result.interrupts_per_sec, (int, float)) and result.interrupts_per_sec >= 0
    assert isinstance(result.processes_running, int) and result.processes_running >= 0
    assert isinstance(result.processes_blocked, int) and result.processes_blocked >= 0
    
    # Logical relationships - both should be non-negative
    assert result.processes_running >= 0 and result.processes_blocked >= 0


@pytest.mark.asyncio
async def test_get_enhanced_memory_info_contract():
    """Test enhanced memory info tool returns valid EnhancedMemoryInfo model"""
    result = await get_enhanced_memory_info()
    
    # Must be EnhancedMemoryInfo instance
    assert isinstance(result, EnhancedMemoryInfo)
    
    # Required fields must exist and be reasonable
    assert isinstance(result.total_gb, (int, float)) and result.total_gb > 0
    assert isinstance(result.available_gb, (int, float)) and result.available_gb >= 0
    assert isinstance(result.used_gb, (int, float)) and result.used_gb >= 0
    assert isinstance(result.usage_percent, (int, float)) and 0 <= result.usage_percent <= 100
    assert isinstance(result.swap_total_gb, (int, float)) and result.swap_total_gb >= 0
    assert isinstance(result.swap_used_gb, (int, float)) and result.swap_used_gb >= 0
    
    # Optional enhanced fields
    if result.buffers_gb is not None:
        assert isinstance(result.buffers_gb, (int, float)) and result.buffers_gb >= 0
    if result.cached_gb is not None:
        assert isinstance(result.cached_gb, (int, float)) and result.cached_gb >= 0
    if result.shared_gb is not None:
        assert isinstance(result.shared_gb, (int, float)) and result.shared_gb >= 0
    
    # Logical relationships
    assert result.used_gb <= result.total_gb
    assert result.swap_used_gb <= result.swap_total_gb


@pytest.mark.asyncio
async def test_get_enhanced_network_stats_contract():
    """Test enhanced network stats tool returns valid list of EnhancedNetworkInfo models"""
    result = await get_enhanced_network_stats()
    
    # Must be a list of EnhancedNetworkInfo instances
    assert isinstance(result, list)
    assert len(result) > 0  # Should have at least one network interface
    
    # Each item must be EnhancedNetworkInfo
    for interface in result:
        assert isinstance(interface, EnhancedNetworkInfo)
        
        # Required fields must exist and be reasonable
        assert isinstance(interface.interface_name, str) and len(interface.interface_name) > 0
        assert isinstance(interface.bytes_sent, int) and interface.bytes_sent >= 0
        assert isinstance(interface.bytes_recv, int) and interface.bytes_recv >= 0
        assert isinstance(interface.packets_sent, int) and interface.packets_sent >= 0
        assert isinstance(interface.packets_recv, int) and interface.packets_recv >= 0
        assert isinstance(interface.bytes_sent_per_sec, (int, float)) and interface.bytes_sent_per_sec >= 0
        assert isinstance(interface.bytes_recv_per_sec, (int, float)) and interface.bytes_recv_per_sec >= 0
        assert isinstance(interface.packets_sent_per_sec, (int, float)) and interface.packets_sent_per_sec >= 0
        assert isinstance(interface.packets_recv_per_sec, (int, float)) and interface.packets_recv_per_sec >= 0
        assert isinstance(interface.is_up, bool)


@pytest.mark.asyncio
async def test_get_performance_snapshot_contract():
    """Test performance snapshot tool returns valid SystemPerformanceSnapshot model"""
    result = await get_performance_snapshot()
    
    # Must be SystemPerformanceSnapshot instance
    assert isinstance(result, SystemPerformanceSnapshot)
    
    # Required fields must exist and be valid models
    assert isinstance(result.io_performance, IOPerformanceInfo)
    assert isinstance(result.system_load, SystemLoadInfo)
    assert isinstance(result.enhanced_memory, EnhancedMemoryInfo)
    assert isinstance(result.enhanced_network, list)
    for interface in result.enhanced_network:
        assert isinstance(interface, EnhancedNetworkInfo)
    assert isinstance(result.collection_time, datetime)
    
    # Collection time should be recent
    now = datetime.now()
    time_diff = abs((now - result.collection_time).total_seconds())
    assert time_diff < 60  # Within 1 minute


@pytest.mark.asyncio
async def test_monitor_io_performance_contract():
    """Test I/O performance monitoring tool returns valid data"""
    result = await monitor_io_performance(duration_seconds=2)
    
    # Must be a dictionary with specific structure
    assert isinstance(result, dict)
    required_keys = ["duration_seconds", "io_rates", "busy_time", "samples"]
    assert all(key in result for key in required_keys)
    
    # Values must be numeric or valid types
    assert isinstance(result["samples"], list)
    assert isinstance(result["io_rates"], dict)
    assert isinstance(result["busy_time"], dict)
    assert isinstance(result["duration_seconds"], int) and result["duration_seconds"] == 2
    
    # Check io_rates structure
    assert "average_bytes_per_sec" in result["io_rates"]
    assert "minimum_bytes_per_sec" in result["io_rates"]
    assert "maximum_bytes_per_sec" in result["io_rates"]
    
    # Check busy_time structure
    assert "average_percent" in result["busy_time"]
    assert "minimum_percent" in result["busy_time"]
    assert "maximum_percent" in result["busy_time"]
    
    # Should have expected number of samples
    assert len(result["samples"]) == 2
    
    # Each sample should be a dict with required fields
    for sample in result["samples"]:
        assert isinstance(sample, dict)
        assert "timestamp" in sample
        assert "read_bytes_per_sec" in sample
        assert "write_bytes_per_sec" in sample
        assert "total_bytes_per_sec" in sample
        assert "busy_time_percent" in sample


# =============================================================================
# RESOURCE CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_live_cpu_resource_contract():
    """Test CPU resource returns formatted string"""
    result = await live_cpu_resource()
    
    # Must be a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should contain basic CPU information indicators
    # Don't assume exact format, just check for key indicators
    result_lower = result.lower()
    assert any(word in result_lower for word in ['cpu', 'usage', 'cores', 'frequency'])


@pytest.mark.asyncio
async def test_live_memory_resource_contract():
    """Test memory resource returns formatted string"""
    result = await live_memory_resource()
    
    # Must be a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should contain memory information indicators
    result_lower = result.lower()
    assert any(word in result_lower for word in ['memory', 'gb', 'ram'])


@pytest.mark.asyncio
async def test_system_config_resource_contract():
    """Test system config resource returns formatted string"""
    result = await system_config_resource()
    
    # Must be a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should contain system configuration indicators
    result_lower = result.lower()
    assert any(word in result_lower for word in ['system', 'configuration', 'os', 'cpu', 'hostname'])


@pytest.mark.asyncio
async def test_live_io_performance_resource_contract():
    """Test I/O performance resource returns formatted string"""
    result = await live_io_performance_resource()
    
    # Must be a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should contain I/O performance indicators
    result_lower = result.lower()
    assert any(word in result_lower for word in ['i/o', 'performance', 'read', 'write', 'bytes', 'disk'])


@pytest.mark.asyncio
async def test_live_system_load_resource_contract():
    """Test system load resource returns formatted string"""
    result = await live_system_load_resource()
    
    # Must be a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should contain system load indicators
    result_lower = result.lower()
    assert any(word in result_lower for word in ['load', 'average', 'processes', 'cpu'])


@pytest.mark.asyncio
async def test_live_network_performance_resource_contract():
    """Test network performance resource returns formatted string"""
    result = await live_network_performance_resource()
    
    # Must be a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should contain network performance indicators
    result_lower = result.lower()
    assert any(word in result_lower for word in ['network', 'performance', 'bytes', 'packets', 'connections'])


# =============================================================================
# BEHAVIORAL TESTS (ARCHITECTURE AGNOSTIC)
# =============================================================================

@pytest.mark.asyncio
async def test_data_types_consistent():
    """Test that data types are consistent across calls"""
    # Get data twice
    cpu1 = await get_cpu_info()
    cpu2 = await get_cpu_info()
    
    # Types should be identical
    assert type(cpu1.usage_percent) == type(cpu2.usage_percent)
    assert type(cpu1.core_count) == type(cpu2.core_count)
    assert type(cpu1.processor_name) == type(cpu2.processor_name)
    
    # Static data should be identical
    assert cpu1.core_count == cpu2.core_count
    assert cpu1.thread_count == cpu2.thread_count
    assert cpu1.processor_name == cpu2.processor_name


@pytest.mark.asyncio
async def test_resource_strings_non_empty():
    """Test that all resource strings are meaningful"""
    cpu_resource = await live_cpu_resource()
    memory_resource = await live_memory_resource()
    config_resource = await system_config_resource()
    
    # Phase 1 resources
    io_resource = await live_io_performance_resource()
    load_resource = await live_system_load_resource()
    network_resource = await live_network_performance_resource()
    
    # All should be non-empty strings
    assert isinstance(cpu_resource, str) and len(cpu_resource.strip()) > 10
    assert isinstance(memory_resource, str) and len(memory_resource.strip()) > 10
    assert isinstance(config_resource, str) and len(config_resource.strip()) > 10
    assert isinstance(io_resource, str) and len(io_resource.strip()) > 10
    assert isinstance(load_resource, str) and len(load_resource.strip()) > 10
    assert isinstance(network_resource, str) and len(network_resource.strip()) > 10


@pytest.mark.asyncio
async def test_concurrent_calls_safe():
    """Test that concurrent calls don't interfere with each other"""
    # Run multiple operations concurrently including Phase 1 tools
    tasks = [
        get_current_datetime(),
        get_cpu_info(),
        get_memory_info(),
        get_io_performance(),
        get_system_load(),
        live_cpu_resource(),
        live_memory_resource(),
        live_network_performance_resource()
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verify all results have correct types
    assert isinstance(results[0], str)  # datetime
    assert isinstance(results[1], CPUInfo)
    assert isinstance(results[2], MemoryInfo)
    assert isinstance(results[3], IOPerformanceInfo)
    assert isinstance(results[4], SystemLoadInfo)
    assert isinstance(results[5], str)  # cpu resource
    assert isinstance(results[6], str)  # memory resource
    assert isinstance(results[7], str)  # io resource


@pytest.mark.asyncio
async def test_error_handling_robust():
    """Test that functions handle errors gracefully"""
    # These should never raise unhandled exceptions
    try:
        # Test with edge case parameters
        processes = await get_top_processes(limit=0)
        assert isinstance(processes, list)
        
        # Test monitoring with minimal duration
        monitoring = await monitor_cpu_usage(duration_seconds=1)
        assert isinstance(monitoring, dict)
        
    except Exception as e:
        pytest.fail(f"Functions should handle edge cases gracefully: {e}")


@pytest.mark.asyncio
async def test_data_freshness():
    """Test that data is reasonably fresh"""
    # Get current time and system data
    before = datetime.now()
    snapshot = await get_system_snapshot()
    after = datetime.now()
    
    # Collection time should be recent (within 10 seconds)
    time_diff = abs((before - snapshot.collection_time).total_seconds())
    assert time_diff < 10, f"Collection time should be recent, but was {time_diff} seconds ago"
    
    # System timestamp should be recent (within 10 seconds)
    time_diff = abs((before - snapshot.system.timestamp).total_seconds())
    assert time_diff < 10, f"System timestamp should be recent, but was {time_diff} seconds ago"


# =============================================================================
# INTEGRATION TESTS (GENERIC)
# =============================================================================

@pytest.mark.asyncio
async def test_snapshot_completeness():
    """Test that snapshot contains complete system information"""
    snapshot = await get_system_snapshot()
    
    # Should have data for all major components
    assert snapshot.cpu.core_count > 0
    assert snapshot.memory.total_gb > 0
    assert len(snapshot.disks) > 0
    # GPUs may be empty list, that's ok
    assert isinstance(snapshot.gpus, list)
    
    # All disks should have reasonable data
    total_disk_space = sum(disk.total_gb for disk in snapshot.disks)
    assert total_disk_space > 0


@pytest.mark.asyncio
async def test_monitoring_consistency():
    """Test that monitoring data is internally consistent"""
    # Monitor for short period
    result = await monitor_cpu_usage(duration_seconds=2)
    
    # Should have expected number of samples
    assert len(result["samples"]) == 2
    
    # Statistics should make sense
    samples = result["samples"]
    calculated_min = min(samples)
    calculated_max = max(samples)
    calculated_avg = sum(samples) / len(samples)
    
    # Allow for small floating point differences
    assert abs(result["minimum"] - calculated_min) < 0.1
    assert abs(result["maximum"] - calculated_max) < 0.1
    assert abs(result["average"] - calculated_avg) < 0.1


@pytest.mark.asyncio
async def test_all_tools_return_expected_types():
    """Comprehensive test that all MCP tools return expected types"""
    
    # Test all tools and their expected return types
    tool_tests = [
        (get_current_datetime, str),
        (get_cpu_info, CPUInfo),
        (get_gpu_info, list),
        (get_memory_info, MemoryInfo),
        (get_disk_info, list),
        (get_system_snapshot, SystemSnapshot),
        (get_network_stats, dict),
        # Phase 1 tools
        (get_io_performance, IOPerformanceInfo),
        (get_system_load, SystemLoadInfo),
        (get_enhanced_memory_info, EnhancedMemoryInfo),
        (get_enhanced_network_stats, list),
        (get_performance_snapshot, SystemPerformanceSnapshot),
    ]
    
    for tool_func, expected_type in tool_tests:
        result = await tool_func()
        assert isinstance(result, expected_type), f"{tool_func.__name__} should return {expected_type}"


@pytest.mark.asyncio
async def test_all_resources_return_strings():
    """Test that all MCP resources return non-empty strings"""
    
    resource_funcs = [
        live_cpu_resource,
        live_memory_resource,
        system_config_resource,
        # Phase 1 resources
        live_io_performance_resource,
        live_system_load_resource,
        live_network_performance_resource
    ]
    
    for resource_func in resource_funcs:
        result = await resource_func()
        assert isinstance(result, str), f"{resource_func.__name__} should return string"
        assert len(result.strip()) > 0, f"{resource_func.__name__} should return non-empty string"


# =============================================================================
# PERFORMANCE TESTS (GENERIC)
# =============================================================================

@pytest.mark.asyncio
async def test_reasonable_response_times():
    """Test that response times are reasonable (no specific thresholds)"""
    import time
    
    # Test individual tools including Phase 1
    tools_to_test = [
        get_current_datetime,
        get_cpu_info,
        get_memory_info,
        get_io_performance,
        get_system_load,
        live_cpu_resource,
        live_memory_resource,
        live_network_performance_resource
    ]
    
    for tool_func in tools_to_test:
        start_time = time.time()
        await tool_func()
        duration = time.time() - start_time
        
        # Very generous timeout - just ensure it doesn't hang
        assert duration < 60.0, f"{tool_func.__name__} took {duration:.2f}s (too long)"


@pytest.mark.asyncio  
@pytest.mark.slow
async def test_system_snapshot_performance():
    """Test that system snapshot completes in reasonable time"""
    import time
    
    start_time = time.time()
    snapshot = await get_system_snapshot()
    duration = time.time() - start_time
    
    # Very generous timeout for any architecture
    assert duration < 120.0, f"System snapshot took {duration:.2f}s (too long)"
    assert isinstance(snapshot, SystemSnapshot)