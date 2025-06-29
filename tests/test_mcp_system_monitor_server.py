import pytest
from unittest.mock import Mock, patch, AsyncMock
from mcp_system_monitor_server import (CPUInfo, CPUCollector, GPUInfo, GPUCollector, DiskInfo, DiskCollector, SystemInfo, SystemSnapshot,
                                       SystemCollector, NetworkCollector, MemoryCollector, MemoryInfo, ProcessCollector)


@pytest.mark.asyncio
async def test_cpu_collector():
    collector = CPUCollector()
    data = await collector.collect_data()
    assert data is not None
    # Validate with Pydantic model
    cpu_info = CPUInfo(**data)
    assert isinstance(cpu_info, CPUInfo)
    assert cpu_info.usage_percent >= 0
    assert len(cpu_info.usage_per_core) > 0


@pytest.mark.asyncio
async def test_gpu_collector():
    collector = GPUCollector()
    data = await collector.collect_data()
    assert data is not None
    assert "gpus" in data
    # Validate with Pydantic model if GPUs are present
    if data["gpus"]:
        gpu_info = [GPUInfo(**gpu) for gpu in data["gpus"]]
        assert all(isinstance(info, GPUInfo) for info in gpu_info)


@pytest.mark.asyncio
async def test_memory_collector():
    collector = MemoryCollector()
    data = await collector.collect_data()
    assert data is not None
    # Validate with Pydantic model
    memory_info = MemoryInfo(**data)
    assert isinstance(memory_info, MemoryInfo)
    assert memory_info.total_gb > 0


@pytest.mark.asyncio
async def test_disk_collector():
    collector = DiskCollector()
    data = await collector.collect_data()
    assert data is not None
    assert "disks" in data
    # Validate with Pydantic model if disks are present
    if data["disks"]:
        disk_info = [DiskInfo(**disk) for disk in data["disks"]]
        assert all(isinstance(info, DiskInfo) for info in disk_info)


@pytest.mark.asyncio
async def test_system_collector():
    collector = SystemCollector()
    data = await collector.collect_data()
    assert data is not None
    # Validate with Pydantic model
    system_info = SystemInfo(**data)
    assert isinstance(system_info, SystemInfo)
    assert system_info.uptime_seconds > 0


@pytest.mark.asyncio
async def test_process_collector():
    collector = ProcessCollector()
    data = await collector.collect_data()
    assert data is not None
    assert "processes" in data
    top_processes = await collector.get_top_processes()
    assert isinstance(top_processes, list)


@pytest.mark.asyncio
async def test_network_collector():
    collector = NetworkCollector()
    data = await collector.collect_data()
    assert data is not None
    assert "interfaces" in data


@pytest.mark.asyncio
async def test_system_snapshot():
    """Test complete system snapshot collection"""
    from mcp_system_monitor_server import get_system_snapshot

    snapshot = await get_system_snapshot()
    assert isinstance(snapshot, SystemSnapshot)
    assert isinstance(snapshot.system, SystemInfo)
    assert isinstance(snapshot.cpu, CPUInfo)
    assert isinstance(snapshot.memory, MemoryInfo)
    assert all(isinstance(disk, DiskInfo) for disk in snapshot.disks)


@pytest.mark.asyncio
async def test_monitor_cpu_usage():
    """Test CPU monitoring over time"""
    from mcp_system_monitor_server import monitor_cpu_usage

    result = await monitor_cpu_usage(duration_seconds=2)
    assert "average" in result
    assert "minimum" in result
    assert "maximum" in result
    assert "samples" in result
    assert len(result["samples"]) == 2


@pytest.mark.asyncio
async def test_get_top_processes():
    """Test top processes retrieval"""
    from mcp_system_monitor_server import process_collector

    # Test CPU sorting
    cpu_processes = await process_collector.get_top_processes(limit=5, sort_by='cpu_percent')
    assert isinstance(cpu_processes, list)
    assert len(cpu_processes) <= 5

    # Test memory sorting
    mem_processes = await process_collector.get_top_processes(limit=5, sort_by='memory_percent')
    assert isinstance(mem_processes, list)
    assert len(mem_processes) <= 5


@pytest.mark.asyncio
async def test_collector_caching():
    """Test that collectors properly cache data"""
    collector = CPUCollector()

    # First call should collect fresh data
    data1 = await collector.get_cached_data(max_age=0.1)

    # Immediate second call should return cached data
    data2 = await collector.get_cached_data(max_age=10.0)
    assert data1 == data2

    # Wait and force fresh data
    import asyncio
    await asyncio.sleep(0.2)
    data3 = await collector.get_cached_data(max_age=0.1)
    # CPU usage should have changed
    assert data3["usage_percent"] != data1["usage_percent"] or data3["usage_per_core"] != data1["usage_per_core"]


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in collectors"""
    with patch('psutil.cpu_percent', side_effect=Exception("Test error")):
        collector = CPUCollector()
        with pytest.raises(Exception):
            await collector.collect_data()


@pytest.mark.asyncio
async def test_gpu_collector_no_gpu():
    """Test GPU collector when no GPUs are available"""
    with patch('mcp_system_monitor_server.PYNVML_AVAILABLE', False):
        with patch('mcp_system_monitor_server.NVML_AVAILABLE', False):
            collector = GPUCollector()
            data = await collector.collect_data()
            assert data["gpus"] == []


@pytest.mark.asyncio
async def test_disk_collector_permission_error():
    """Test disk collector handling permission errors"""
    with patch('psutil.disk_usage', side_effect=PermissionError("Access denied")):
        collector = DiskCollector()
        data = await collector.collect_data()
        # Should handle error gracefully and return empty or partial data
        assert "disks" in data
