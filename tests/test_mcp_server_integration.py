"""
Integration tests for MCP server functionality.
Tests the actual MCP server endpoints and protocol compliance.
"""
import pytest
from unittest.mock import patch, AsyncMock
from mcp.server.fastmcp import FastMCP
from mcp_system_monitor_server import mcp


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_instance():
    """Test that the MCP server instance is properly configured."""
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "SystemMonitor"
    
    # Check dependencies are declared
    expected_deps = ["psutil", "nvidia-ml-py", "pydantic", "pynvml"]
    for dep in expected_deps:
        assert dep in mcp.dependencies


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_tools_registration():
    """Test that all MCP tools are properly registered."""
    # Test by attempting to call each expected tool function
    expected_tools = [
        'get_current_datetime',
        'get_cpu_info', 
        'get_gpu_info',
        'get_memory_info',
        'get_disk_info',
        'get_system_snapshot',
        'monitor_cpu_usage',
        'get_top_processes',
        'get_network_stats'
    ]
    
    # Import the tools directly from the module
    import mcp_system_monitor_server
    
    for tool_name in expected_tools:
        assert hasattr(mcp_system_monitor_server, tool_name), f"Tool {tool_name} not found in module"
        tool_func = getattr(mcp_system_monitor_server, tool_name)
        assert callable(tool_func), f"Tool {tool_name} is not callable"
        assert hasattr(tool_func, '__doc__'), f"Tool {tool_name} missing docstring"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_resources_registration():
    """Test that all MCP resources are properly registered."""
    # Test by attempting to call each expected resource function
    expected_resources = [
        'live_cpu_resource',
        'live_memory_resource', 
        'system_config_resource'
    ]
    
    # Import the resources directly from the module
    import mcp_system_monitor_server
    
    for resource_name in expected_resources:
        assert hasattr(mcp_system_monitor_server, resource_name), f"Resource {resource_name} not found in module"
        resource_func = getattr(mcp_system_monitor_server, resource_name)
        assert callable(resource_func), f"Resource {resource_name} is not callable"
        assert hasattr(resource_func, '__doc__'), f"Resource {resource_name} missing docstring"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_input_validation():
    """Test that tools properly validate their inputs."""
    # Test monitor_cpu_usage with valid input
    from mcp_system_monitor_server import monitor_cpu_usage
    result = await monitor_cpu_usage(duration_seconds=1)
    assert isinstance(result, dict)
    
    # Test get_top_processes with valid input
    from mcp_system_monitor_server import get_top_processes
    result = await get_top_processes(limit=5, sort_by='cpu_percent')
    assert isinstance(result, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_return_types():
    """Test that tools return expected types according to their annotations."""
    from mcp_system_monitor_server import (
        get_current_datetime, get_cpu_info, get_gpu_info, get_memory_info,
        get_disk_info, get_system_snapshot, monitor_cpu_usage, 
        get_top_processes, get_network_stats
    )
    from mcp_system_monitor_server import CPUInfo, MemoryInfo, SystemSnapshot
    from typing import Dict, Any, List
    
    # Test return types match annotations
    datetime_result = await get_current_datetime()
    assert isinstance(datetime_result, str)
    
    cpu_result = await get_cpu_info()
    assert isinstance(cpu_result, CPUInfo)
    
    gpu_result = await get_gpu_info()
    assert isinstance(gpu_result, list)
    
    memory_result = await get_memory_info()
    assert isinstance(memory_result, MemoryInfo)
    
    disk_result = await get_disk_info()
    assert isinstance(disk_result, list)
    
    snapshot_result = await get_system_snapshot()
    assert isinstance(snapshot_result, SystemSnapshot)
    
    monitor_result = await monitor_cpu_usage(1)
    assert isinstance(monitor_result, dict)
    
    processes_result = await get_top_processes()
    assert isinstance(processes_result, list)
    
    network_result = await get_network_stats()
    assert isinstance(network_result, dict)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resource_return_types():
    """Test that resources return expected string types."""
    from mcp_system_monitor_server import (
        live_cpu_resource, live_memory_resource, system_config_resource
    )
    
    cpu_resource = await live_cpu_resource()
    assert isinstance(cpu_resource, str)
    assert len(cpu_resource) > 0
    
    memory_resource = await live_memory_resource()
    assert isinstance(memory_resource, str) 
    assert len(memory_resource) > 0
    
    config_resource = await system_config_resource()
    assert isinstance(config_resource, str)
    assert len(config_resource) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_server_capabilities():
    """Test that the server properly declares its capabilities."""
    # Test that the server instance exists and is properly configured
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "SystemMonitor"
    
    # Test that we can call the tools and resources
    from mcp_system_monitor_server import get_current_datetime, live_cpu_resource
    
    # These should be callable without errors
    datetime_result = await get_current_datetime()
    assert isinstance(datetime_result, str)
    
    cpu_resource_result = await live_cpu_resource()
    assert isinstance(cpu_resource_result, str)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_propagation():
    """Test that errors in tools are properly handled and propagated."""
    from mcp_system_monitor_server import get_top_processes
    
    # Test with invalid sort_by parameter
    try:
        result = await get_top_processes(sort_by='invalid_field')
        # Should handle gracefully and return empty or filtered results
        assert isinstance(result, list)
    except Exception as e:
        # If it raises an exception, it should be informative
        assert str(e) != ""


@pytest.mark.integration
@pytest.mark.asyncio 
async def test_concurrent_tool_execution():
    """Test that multiple tools can be executed concurrently."""
    import asyncio
    from mcp_system_monitor_server import (
        get_current_datetime, get_cpu_info, get_memory_info, CPUInfo, MemoryInfo
    )
    
    # Execute multiple tools concurrently
    tasks = [
        get_current_datetime(),
        get_cpu_info(),
        get_memory_info(),
        get_current_datetime(),
        get_cpu_info()
    ]
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    assert isinstance(results[0], str)  # datetime
    assert isinstance(results[1], CPUInfo) 
    assert isinstance(results[2], MemoryInfo)
    assert isinstance(results[3], str)  # datetime
    assert isinstance(results[4], CPUInfo)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resource_caching_behavior():
    """Test that resources utilize caching appropriately."""
    from mcp_system_monitor_server import live_cpu_resource, system_config_resource
    
    # Config resource should return mostly identical results (static data)
    # Note: uptime will change, so we check everything except the uptime line
    config1 = await system_config_resource()
    config2 = await system_config_resource()
    
    # Split by lines and check that most lines are identical
    lines1 = config1.split('\n')
    lines2 = config2.split('\n')
    
    # Should have same number of lines
    assert len(lines1) == len(lines2)
    
    # All lines except uptime should be identical
    non_uptime_lines1 = [line for line in lines1 if not line.startswith('Uptime:')]
    non_uptime_lines2 = [line for line in lines2 if not line.startswith('Uptime:')]
    assert non_uptime_lines1 == non_uptime_lines2
    
    # CPU resource may vary but should have consistent format
    cpu1 = await live_cpu_resource()
    cpu2 = await live_cpu_resource()
    
    # Both should have the expected format elements
    assert "Usage:" in cpu1 and "Usage:" in cpu2
    assert "Cores:" in cpu1 and "Cores:" in cpu2
    assert "Freq:" in cpu1 and "Freq:" in cpu2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_model_validation():
    """Test that Pydantic models properly validate data."""
    from mcp_system_monitor_server import get_cpu_info, get_memory_info, get_system_snapshot
    
    # Get data and ensure it validates through Pydantic models
    cpu_info = await get_cpu_info()
    
    # Test that the model has all required fields
    assert hasattr(cpu_info, 'usage_percent')
    assert hasattr(cpu_info, 'core_count') 
    assert hasattr(cpu_info, 'processor_name')
    
    # Test field constraints
    assert 0 <= cpu_info.usage_percent <= 100
    assert cpu_info.core_count > 0
    
    memory_info = await get_memory_info()
    assert memory_info.total_gb > 0
    assert 0 <= memory_info.usage_percent <= 100
    
    # Test snapshot contains all expected components
    snapshot = await get_system_snapshot()
    assert snapshot.system is not None
    assert snapshot.cpu is not None
    assert snapshot.memory is not None
    assert snapshot.gpus is not None  # List, may be empty
    assert snapshot.disks is not None  # List, should have at least one


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_monitoring_tool_functionality():
    """Test the monitoring tools work as expected."""
    from mcp_system_monitor_server import monitor_cpu_usage
    
    # Test actual monitoring functionality
    result = await monitor_cpu_usage(duration_seconds=2)
    
    assert "samples" in result
    assert "average" in result
    assert "minimum" in result
    assert "maximum" in result
    
    assert len(result["samples"]) == 2
    assert result["minimum"] <= result["average"] <= result["maximum"]
    
    # All values should be reasonable percentages
    assert 0 <= result["minimum"] <= 100
    assert 0 <= result["maximum"] <= 100
    assert 0 <= result["average"] <= 100


@pytest.mark.integration
@pytest.mark.asyncio
async def test_system_snapshot_completeness():
    """Test that system snapshot captures all system components."""
    from mcp_system_monitor_server import get_system_snapshot
    
    snapshot = await get_system_snapshot()
    
    # Verify all major components are captured
    assert snapshot.system.hostname is not None
    assert snapshot.system.platform is not None
    assert snapshot.system.uptime_seconds >= 0
    
    assert snapshot.cpu.core_count > 0
    assert snapshot.cpu.processor_name is not None
    
    assert snapshot.memory.total_gb > 0
    
    # Should have at least one disk
    assert len(snapshot.disks) > 0
    assert all(disk.total_gb > 0 for disk in snapshot.disks)
    
    # GPUs may or may not be present
    assert isinstance(snapshot.gpus, list)