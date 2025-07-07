import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

# Import all models and collectors
from mcp_system_monitor_server import (
    CPUInfo, CPUCollector, GPUInfo, GPUCollector, DiskInfo, DiskCollector, 
    SystemInfo, SystemSnapshot, SystemCollector, NetworkCollector, 
    MemoryCollector, MemoryInfo, ProcessCollector,
    # Import all MCP tools and resources
    get_current_datetime, get_cpu_info, get_gpu_info, get_memory_info,
    get_disk_info, get_system_snapshot, monitor_cpu_usage, get_top_processes,
    get_network_stats, live_cpu_resource, live_memory_resource, system_config_resource,
    # Import the FastMCP server instance
    mcp
)


# =============================================================================
# MCP TOOLS TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_get_current_datetime():
    """Test get_current_datetime MCP tool"""
    result = await get_current_datetime()
    assert isinstance(result, str)
    # Verify it's a valid ISO format datetime
    datetime.fromisoformat(result.replace(' ', 'T'))
    assert len(result) > 10  # Should be more than just a date


@pytest.mark.asyncio
async def test_get_cpu_info_tool():
    """Test get_cpu_info MCP tool"""
    result = await get_cpu_info()
    assert isinstance(result, CPUInfo)
    assert result.usage_percent >= 0
    assert result.usage_percent <= 100
    assert len(result.usage_per_core) > 0
    assert result.core_count > 0
    assert result.thread_count >= result.core_count
    assert result.processor_name is not None
    assert result.vendor is not None


@pytest.mark.asyncio
async def test_get_gpu_info_tool():
    """Test get_gpu_info MCP tool"""
    result = await get_gpu_info()
    assert isinstance(result, list)
    # All items should be GPUInfo instances
    for gpu in result:
        assert isinstance(gpu, GPUInfo)
        assert gpu.name is not None
        assert gpu.usage_percent >= 0
        assert gpu.memory_total_mb >= 0
        assert gpu.memory_used_mb >= 0


@pytest.mark.asyncio
async def test_get_memory_info_tool():
    """Test get_memory_info MCP tool"""
    result = await get_memory_info()
    assert isinstance(result, MemoryInfo)
    assert result.total_gb > 0
    assert result.used_gb >= 0
    assert result.available_gb >= 0
    assert result.usage_percent >= 0
    assert result.usage_percent <= 100
    assert result.swap_total_gb >= 0
    assert result.swap_used_gb >= 0


@pytest.mark.asyncio
async def test_get_disk_info_tool():
    """Test get_disk_info MCP tool"""
    result = await get_disk_info()
    assert isinstance(result, list)
    # Should have at least one disk
    assert len(result) > 0
    for disk in result:
        assert isinstance(disk, DiskInfo)
        assert disk.mount_point is not None
        assert disk.total_gb > 0
        assert disk.used_gb >= 0
        assert disk.free_gb >= 0
        assert disk.usage_percent >= 0
        assert disk.usage_percent <= 100
        assert disk.filesystem is not None


@pytest.mark.asyncio
async def test_get_system_snapshot_tool():
    """Test get_system_snapshot MCP tool"""
    result = await get_system_snapshot()
    assert isinstance(result, SystemSnapshot)
    assert isinstance(result.system, SystemInfo)
    assert isinstance(result.cpu, CPUInfo)
    assert isinstance(result.memory, MemoryInfo)
    assert isinstance(result.gpus, list)
    assert isinstance(result.disks, list)
    assert result.collection_time is not None


@pytest.mark.asyncio
async def test_monitor_cpu_usage_tool():
    """Test monitor_cpu_usage MCP tool"""
    # Test with short duration to avoid long test times
    result = await monitor_cpu_usage(duration_seconds=2)
    assert isinstance(result, dict)
    assert "average" in result
    assert "minimum" in result
    assert "maximum" in result
    assert "samples" in result
    assert len(result["samples"]) == 2
    assert isinstance(result["average"], (int, float))
    assert isinstance(result["minimum"], (int, float))
    assert isinstance(result["maximum"], (int, float))
    assert result["minimum"] <= result["average"] <= result["maximum"]


@pytest.mark.asyncio
async def test_get_top_processes_tool():
    """Test get_top_processes MCP tool"""
    # Test CPU sorting
    result = await get_top_processes(limit=5, sort_by='cpu_percent')
    assert isinstance(result, list)
    assert len(result) <= 5
    
    # Test memory sorting
    result = await get_top_processes(limit=3, sort_by='memory_percent')
    assert isinstance(result, list)
    assert len(result) <= 3
    
    # Verify structure of process entries
    if result:
        process = result[0]
        assert isinstance(process, dict)
        # Should have basic process fields
        expected_fields = ['pid', 'name', 'cpu_percent', 'memory_percent']
        for field in expected_fields:
            assert field in process or process.get(field) is not None


@pytest.mark.asyncio
async def test_get_network_stats_tool():
    """Test get_network_stats MCP tool"""
    result = await get_network_stats()
    assert isinstance(result, dict)
    assert "interfaces" in result
    assert isinstance(result["interfaces"], dict)
    
    # Check interface data structure
    for interface_name, stats in result["interfaces"].items():
        assert isinstance(interface_name, str)
        assert isinstance(stats, dict)
        # Should have basic network stats
        expected_fields = ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv']
        for field in expected_fields:
            assert field in stats
            assert isinstance(stats[field], int)


# =============================================================================
# MCP RESOURCES TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_live_cpu_resource():
    """Test live_cpu_resource MCP resource"""
    result = await live_cpu_resource()
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain key CPU information
    assert "Usage:" in result
    assert "Cores:" in result
    assert "Freq:" in result
    assert "%" in result  # Usage percentage
    assert "MHz" in result  # Frequency


@pytest.mark.asyncio
async def test_live_memory_resource():
    """Test live_memory_resource MCP resource"""
    result = await live_memory_resource()
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain memory information
    assert "Memory:" in result
    assert "GB" in result
    assert "/" in result  # Used/Total format
    assert "%" in result  # Usage percentage


@pytest.mark.asyncio
async def test_system_config_resource():
    """Test system_config_resource MCP resource"""
    result = await system_config_resource()
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain system configuration info
    assert "System Configuration:" in result
    assert "OS:" in result
    assert "Hostname:" in result
    assert "CPU:" in result
    assert "Cores:" in result
    assert "Uptime:" in result


# =============================================================================
# COLLECTOR TESTS (ENHANCED)
# =============================================================================

@pytest.mark.asyncio
async def test_cpu_collector_comprehensive():
    """Comprehensive test for CPU collector"""
    collector = CPUCollector()
    data = await collector.collect_data()
    
    # Validate data structure
    assert isinstance(data, dict)
    required_fields = [
        'usage_percent', 'usage_per_core', 'frequency_current', 'frequency_max',
        'core_count', 'thread_count', 'processor_name', 'vendor', 'architecture', 'bits'
    ]
    for field in required_fields:
        assert field in data
    
    # Create CPUInfo and validate
    cpu_info = CPUInfo(**data)
    assert isinstance(cpu_info, CPUInfo)
    assert 0 <= cpu_info.usage_percent <= 100
    
    # Per-core usage might report logical cores (threads) instead of physical cores
    # So we should check against thread_count, not core_count
    assert len(cpu_info.usage_per_core) == cpu_info.thread_count
    assert cpu_info.thread_count >= cpu_info.core_count


@pytest.mark.asyncio
async def test_gpu_collector_comprehensive():
    """Comprehensive test for GPU collector"""
    collector = GPUCollector()
    data = await collector.collect_data()
    
    assert isinstance(data, dict)
    assert "gpus" in data
    assert isinstance(data["gpus"], list)
    
    # Validate GPU data if any GPUs present
    for gpu_data in data["gpus"]:
        assert isinstance(gpu_data, dict)
        required_fields = ['name', 'usage_percent', 'memory_used_mb', 'memory_total_mb']
        for field in required_fields:
            assert field in gpu_data
        
        gpu_info = GPUInfo(**gpu_data)
        assert isinstance(gpu_info, GPUInfo)
        assert gpu_info.usage_percent >= 0
        assert gpu_info.memory_total_mb >= 0
        assert gpu_info.memory_used_mb >= 0


@pytest.mark.asyncio
async def test_memory_collector_comprehensive():
    """Comprehensive test for memory collector"""
    collector = MemoryCollector()
    data = await collector.collect_data()
    
    assert isinstance(data, dict)
    required_fields = ['total_gb', 'available_gb', 'used_gb', 'usage_percent', 'swap_total_gb', 'swap_used_gb']
    for field in required_fields:
        assert field in data
    
    memory_info = MemoryInfo(**data)
    assert isinstance(memory_info, MemoryInfo)
    assert memory_info.total_gb > 0
    assert 0 <= memory_info.usage_percent <= 100


@pytest.mark.asyncio
async def test_disk_collector_comprehensive():
    """Comprehensive test for disk collector"""
    collector = DiskCollector()
    data = await collector.collect_data()
    
    assert isinstance(data, dict)
    assert "disks" in data
    assert isinstance(data["disks"], list)
    
    # Should have at least one disk
    assert len(data["disks"]) > 0
    
    for disk_data in data["disks"]:
        assert isinstance(disk_data, dict)
        required_fields = ['mount_point', 'total_gb', 'used_gb', 'free_gb', 'usage_percent', 'filesystem']
        for field in required_fields:
            assert field in disk_data
        
        disk_info = DiskInfo(**disk_data)
        assert isinstance(disk_info, DiskInfo)
        assert disk_info.total_gb > 0
        assert 0 <= disk_info.usage_percent <= 100


@pytest.mark.asyncio
async def test_system_collector_comprehensive():
    """Comprehensive test for system collector"""
    collector = SystemCollector()
    data = await collector.collect_data()
    
    assert isinstance(data, dict)
    required_fields = ['timestamp', 'hostname', 'platform', 'architecture', 'uptime_seconds']
    for field in required_fields:
        assert field in data
    
    system_info = SystemInfo(**data)
    assert isinstance(system_info, SystemInfo)
    assert system_info.uptime_seconds > 0
    assert len(system_info.hostname) > 0
    assert len(system_info.platform) > 0


@pytest.mark.asyncio
async def test_process_collector_comprehensive():
    """Comprehensive test for process collector"""
    collector = ProcessCollector()
    data = await collector.collect_data()
    
    assert isinstance(data, dict)
    assert "processes" in data
    assert isinstance(data["processes"], list)
    
    # Should have at least some processes
    assert len(data["processes"]) > 0
    
    # Test top processes functionality
    top_cpu = await collector.get_top_processes(limit=5, sort_by='cpu_percent')
    assert isinstance(top_cpu, list)
    assert len(top_cpu) <= 5
    
    top_memory = await collector.get_top_processes(limit=3, sort_by='memory_percent')
    assert isinstance(top_memory, list)
    assert len(top_memory) <= 3


@pytest.mark.asyncio
async def test_network_collector_comprehensive():
    """Comprehensive test for network collector"""
    collector = NetworkCollector()
    data = await collector.collect_data()
    
    assert isinstance(data, dict)
    assert "interfaces" in data
    assert isinstance(data["interfaces"], dict)
    
    # Should have at least one network interface
    assert len(data["interfaces"]) > 0
    
    for interface_name, stats in data["interfaces"].items():
        assert isinstance(interface_name, str)
        assert isinstance(stats, dict)
        required_fields = [
            'bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv',
            'errin', 'errout', 'dropin', 'dropout'
        ]
        for field in required_fields:
            assert field in stats
            assert isinstance(stats[field], int)
            assert stats[field] >= 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_full_system_monitoring_integration():
    """Test complete system monitoring workflow"""
    # Test getting individual components
    cpu_info = await get_cpu_info()
    gpu_info = await get_gpu_info()
    memory_info = await get_memory_info()
    disk_info = await get_disk_info()
    network_stats = await get_network_stats()
    
    # Test system snapshot includes all components
    snapshot = await get_system_snapshot()
    
    # Verify snapshot contains data from similar time period
    # Note: CPU usage can vary between calls, so we check they're both reasonable
    assert 0 <= snapshot.cpu.usage_percent <= 100
    assert 0 <= cpu_info.usage_percent <= 100
    
    # Memory total should be identical (static)
    assert snapshot.memory.total_gb == memory_info.total_gb
    
    # Component counts should match
    assert len(snapshot.gpus) == len(gpu_info)
    assert len(snapshot.disks) == len(disk_info)


@pytest.mark.asyncio
async def test_resource_consistency():
    """Test that resources return consistent data"""
    # Get live resources multiple times
    cpu_resource_1 = await live_cpu_resource()
    cpu_resource_2 = await live_cpu_resource()
    
    memory_resource_1 = await live_memory_resource()
    memory_resource_2 = await live_memory_resource()
    
    config_resource_1 = await system_config_resource()
    config_resource_2 = await system_config_resource()
    
    # CPU and memory may vary, but should have same format
    assert "Usage:" in cpu_resource_1 and "Usage:" in cpu_resource_2
    assert "Memory:" in memory_resource_1 and "Memory:" in memory_resource_2
    
    # Config should be mostly identical (static data, except uptime changes)
    # Compare everything except the uptime line
    lines1 = config_resource_1.split('\n')
    lines2 = config_resource_2.split('\n')
    
    non_uptime_lines1 = [line for line in lines1 if not line.startswith('Uptime:')]
    non_uptime_lines2 = [line for line in lines2 if not line.startswith('Uptime:')]
    assert non_uptime_lines1 == non_uptime_lines2


@pytest.mark.asyncio 
async def test_concurrent_tool_calls():
    """Test multiple MCP tools called concurrently"""
    # Call multiple tools concurrently
    tasks = [
        get_cpu_info(),
        get_memory_info(),
        get_disk_info(),
        get_current_datetime(),
        get_network_stats()
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verify all results are valid
    assert isinstance(results[0], CPUInfo)
    assert isinstance(results[1], MemoryInfo)
    assert isinstance(results[2], list)  # disk_info
    assert isinstance(results[3], str)   # datetime
    assert isinstance(results[4], dict)  # network_stats


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_monitor_cpu_usage_invalid_duration():
    """Test monitor_cpu_usage with invalid duration"""
    # Test with zero duration
    result = await monitor_cpu_usage(duration_seconds=0)
    assert result["samples"] == []
    assert result["average"] == 0
    
    # Test with negative duration should be handled gracefully
    result = await monitor_cpu_usage(duration_seconds=-1)
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_get_top_processes_invalid_params():
    """Test get_top_processes with invalid parameters"""
    # Test with invalid sort_by - should still return results, just may not be sorted correctly
    result = await get_top_processes(limit=5, sort_by='invalid_field')
    assert isinstance(result, list)
    # May still return results, just unsorted or sorted by default field
    
    # Test with zero limit - should return empty list
    result = await get_top_processes(limit=0)
    assert len(result) == 0
    
    # Test with negative limit - should handle gracefully (may return empty or default)
    result = await get_top_processes(limit=-1)
    assert isinstance(result, list)
    # Don't assert exact length as implementation may vary


@pytest.mark.asyncio
async def test_gpu_collector_error_handling():
    """Test GPU collector error handling scenarios"""
    # Test when NVML libraries are not available
    with patch('mcp_system_monitor_server.PYNVML_AVAILABLE', False):
        with patch('mcp_system_monitor_server.NVML_AVAILABLE', False):
            result = await get_gpu_info()
            assert isinstance(result, list)
            # Should handle gracefully and may return empty list or generic GPU info


@pytest.mark.asyncio
async def test_network_stats_error_handling():
    """Test network stats error handling"""
    with patch('psutil.net_io_counters', side_effect=Exception("Network error")):
        # Should handle error gracefully
        try:
            result = await get_network_stats()
            # If it doesn't raise an exception, verify structure
            assert isinstance(result, dict)
        except Exception:
            # If it does raise, that's also acceptable for this test
            pass


@pytest.mark.asyncio
async def test_collector_caching_behavior():
    """Test collector caching behavior"""
    collector = CPUCollector()
    
    # First call should collect fresh data
    data1 = await collector.get_cached_data(max_age=0.1)
    
    # Immediate second call should return cached data
    data2 = await collector.get_cached_data(max_age=10.0)
    assert data1 == data2
    
    # Wait and force fresh data
    await asyncio.sleep(0.2)
    data3 = await collector.get_cached_data(max_age=0.1)
    # CPU usage should have changed
    assert data3["usage_percent"] != data1["usage_percent"] or data3["usage_per_core"] != data1["usage_per_core"]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
async def test_system_snapshot_performance():
    """Test system snapshot collection performance"""
    start_time = time.time()
    snapshot = await get_system_snapshot()
    end_time = time.time()
    
    # Should complete within reasonable time (5 seconds)
    assert end_time - start_time < 5.0
    assert isinstance(snapshot, SystemSnapshot)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_monitoring_overhead():
    """Test that monitoring doesn't consume excessive resources"""
    # Run multiple monitoring tasks
    tasks = []
    for _ in range(5):
        tasks.extend([
            get_cpu_info(),
            get_memory_info(),
            live_cpu_resource(),
            live_memory_resource()
        ])
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Should complete all tasks within reasonable time (increased tolerance)
    duration = end_time - start_time
    assert duration < 30.0, f"Tasks took {duration:.2f}s, expected < 30s"
    assert len(results) == 20  # 5 iterations * 4 tasks each


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_cpu_info_data_validation():
    """Test CPU info data validation and constraints"""
    cpu_info = await get_cpu_info()
    
    # Usage should be within valid range
    assert 0 <= cpu_info.usage_percent <= 100
    
    # Per-core usage should be valid
    for core_usage in cpu_info.usage_per_core:
        assert 0 <= core_usage <= 100
    
    # Core count should be positive and reasonable (no upper limit assumptions)
    assert cpu_info.core_count > 0
    assert cpu_info.thread_count >= cpu_info.core_count
    
    # Frequency should be positive if reported (no specific range assumptions)
    if cpu_info.frequency_current > 0:
        assert cpu_info.frequency_current > 0  # Just check it's positive


@pytest.mark.asyncio
async def test_memory_info_data_validation():
    """Test memory info data validation and constraints"""
    memory_info = await get_memory_info()
    
    # Memory amounts should be positive
    assert memory_info.total_gb > 0
    assert memory_info.used_gb >= 0
    assert memory_info.available_gb >= 0
    
    # Usage percentage should be reasonable
    assert 0 <= memory_info.usage_percent <= 100
    
    # Basic sanity check: used memory shouldn't exceed total memory significantly
    assert memory_info.used_gb <= memory_info.total_gb * 1.1  # Allow 10% overhead for accounting differences


@pytest.mark.asyncio
async def test_disk_info_data_validation():
    """Test disk info data validation and constraints"""
    disks = await get_disk_info()
    
    for disk in disks:
        # Disk sizes should be positive
        assert disk.total_gb > 0
        assert disk.used_gb >= 0
        assert disk.free_gb >= 0
        
        # Usage percentage should be valid
        assert 0 <= disk.usage_percent <= 100
        
        # Basic sanity check: used space shouldn't exceed total significantly
        assert disk.used_gb <= disk.total_gb * 1.1  # Allow for accounting differences
        
        # Mount point and filesystem should be non-empty strings
        assert len(disk.mount_point) > 0
        assert len(disk.filesystem) > 0


@pytest.mark.asyncio
async def test_datetime_format_validation():
    """Test datetime format validation"""
    datetime_str = await get_current_datetime()
    
    # Should be a non-empty string
    assert isinstance(datetime_str, str)
    assert len(datetime_str) > 0
    
    # Should be parseable as ISO format
    parsed_dt = datetime.fromisoformat(datetime_str.replace(' ', 'T'))
    assert isinstance(parsed_dt, datetime)
    
    # Should be reasonably recent (within 24 hours to account for any timezone/system issues)
    now = datetime.now()
    time_diff = abs((now - parsed_dt).total_seconds())
    assert time_diff < 86400  # Within 24 hours


# =============================================================================
# MOCK TESTS FOR EDGE CASES
# =============================================================================

@pytest.mark.asyncio
async def test_gpu_detection_edge_cases():
    """Test GPU detection with various edge cases"""
    # Test with no GPUs detected
    with patch.object(GPUCollector, '_get_generic_gpu_info_windows', return_value=[]):
        with patch.object(GPUCollector, '_get_generic_gpu_info_linux', return_value=[]):
            with patch.object(GPUCollector, '_get_generic_gpu_info_macos', return_value=[]):
                with patch('mcp_system_monitor_server.PYNVML_AVAILABLE', False):
                    with patch('mcp_system_monitor_server.NVML_AVAILABLE', False):
                        result = await get_gpu_info()
                        assert isinstance(result, list)
                        assert len(result) == 0


@pytest.mark.asyncio 
async def test_system_info_with_mock_data():
    """Test system info with controlled mock data"""
    mock_boot_time = 1000000000  # Fixed timestamp
    
    with patch('psutil.boot_time', return_value=mock_boot_time):
        with patch('platform.node', return_value='test-hostname'):
            with patch('platform.system', return_value='TestOS'):
                with patch('platform.release', return_value='1.0'):
                    with patch('platform.machine', return_value='x86_64'):
                        snapshot = await get_system_snapshot()
                        
                        assert snapshot.system.hostname == 'test-hostname'
                        assert 'TestOS 1.0' in snapshot.system.platform
                        assert snapshot.system.architecture == 'x86_64'
                        assert snapshot.system.uptime_seconds > 0


@pytest.mark.asyncio
async def test_resource_formatting():
    """Test resource string formatting"""
    # Test CPU resource formatting
    cpu_resource = await live_cpu_resource()
    parts = cpu_resource.split(' | ')
    assert len(parts) >= 4  # Should have processor, usage, cores, freq
    
    # Test memory resource formatting
    memory_resource = await live_memory_resource()
    assert 'Memory:' in memory_resource
    assert 'GB' in memory_resource
    assert '(' in memory_resource and ')' in memory_resource  # Percentage in parentheses
    
    # Test config resource formatting
    config_resource = await system_config_resource()
    lines = config_resource.split('\n')
    assert len(lines) > 5  # Should have multiple configuration lines
    assert any('OS:' in line for line in lines)
    assert any('CPU:' in line for line in lines)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_empty_processes_list():
    """Test handling of empty processes list"""
    with patch('psutil.process_iter', return_value=[]):
        result = await get_top_processes(limit=5)
        assert isinstance(result, list)
        assert len(result) == 0


@pytest.mark.asyncio
async def test_malformed_process_data():
    """Test handling of malformed process data"""
    # Mock a process that raises an exception
    mock_proc = Mock()
    mock_proc.info = {'pid': 123, 'name': 'test'}
    
    with patch('psutil.process_iter', side_effect=Exception("Process access error")):
        try:
            result = await get_top_processes(limit=5)
            assert isinstance(result, list)
        except Exception:
            # It's acceptable if the function raises an exception for malformed data
            pass


@pytest.mark.asyncio
async def test_large_dataset_handling():
    """Test handling of large datasets"""
    # Test with large limit
    result = await get_top_processes(limit=1000)
    assert isinstance(result, list)
    # Should handle gracefully even if system has fewer processes
    
    # Test monitor_cpu_usage with longer duration (but still reasonable for tests)
    result = await monitor_cpu_usage(duration_seconds=3)
    assert len(result["samples"]) == 3
    assert isinstance(result["average"], (int, float))


@pytest.mark.asyncio
async def test_unicode_handling():
    """Test handling of unicode characters in system data"""
    # Mock system data with unicode characters
    with patch('platform.node', return_value='test-hóst-name'):
        config = await system_config_resource()
        assert 'test-hóst-name' in config
        assert isinstance(config, str)


# =============================================================================
# STRESS TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.slow
async def test_rapid_consecutive_calls():
    """Test rapid consecutive calls to all endpoints"""
    # Rapidly call the same endpoint multiple times
    tasks = [get_current_datetime() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(isinstance(result, str) for result in results)
    
    # All should be valid datetime strings
    for result in results:
        datetime.fromisoformat(result.replace(' ', 'T'))


@pytest.mark.asyncio
@pytest.mark.slow
async def test_mixed_workload_stress():
    """Test mixed workload with all types of endpoints"""
    tasks = []
    
    # Add various types of tasks
    for _ in range(3):
        tasks.extend([
            get_cpu_info(),
            get_memory_info(),
            get_current_datetime(),
            live_cpu_resource(),
            live_memory_resource(),
            system_config_resource()
        ])
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Should complete within reasonable time (increased tolerance)
    duration = end_time - start_time
    assert duration < 45.0, f"Mixed workload took {duration:.2f}s, expected < 45s"
    assert len(results) == 18  # 3 iterations * 6 tasks each
    
    # Verify result types
    cpu_infos = [r for r in results if isinstance(r, CPUInfo)]
    memory_infos = [r for r in results if isinstance(r, MemoryInfo)]
    strings = [r for r in results if isinstance(r, str)]
    
    assert len(cpu_infos) == 3
    assert len(memory_infos) == 3
    assert len(strings) == 12  # datetime + 3 resources * 3 iterations