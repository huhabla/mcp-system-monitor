"""
Test configuration and fixtures for MCP System Monitor Server tests.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_psutil_cpu():
    """Mock psutil CPU functions with realistic data."""
    with patch('psutil.cpu_percent') as mock_cpu_percent, \
         patch('psutil.cpu_count') as mock_cpu_count, \
         patch('psutil.cpu_freq') as mock_cpu_freq:
        
        mock_cpu_percent.return_value = 45.2
        mock_cpu_count.side_effect = lambda logical=True: 8 if logical else 4
        
        mock_freq = Mock()
        mock_freq.current = 2400.0
        mock_freq.max = 3200.0
        mock_cpu_freq.return_value = mock_freq
        
        yield {
            'cpu_percent': mock_cpu_percent,
            'cpu_count': mock_cpu_count,
            'cpu_freq': mock_cpu_freq
        }


@pytest.fixture
def mock_psutil_memory():
    """Mock psutil memory functions with realistic data."""
    with patch('psutil.virtual_memory') as mock_vmem, \
         patch('psutil.swap_memory') as mock_swap:
        
        mock_vmem_obj = Mock()
        mock_vmem_obj.total = 16 * 1024**3  # 16 GB
        mock_vmem_obj.available = 8 * 1024**3  # 8 GB available
        mock_vmem_obj.used = 8 * 1024**3  # 8 GB used
        mock_vmem_obj.percent = 50.0
        mock_vmem.return_value = mock_vmem_obj
        
        mock_swap_obj = Mock()
        mock_swap_obj.total = 4 * 1024**3  # 4 GB swap
        mock_swap_obj.used = 1 * 1024**3   # 1 GB used
        mock_swap.return_value = mock_swap_obj
        
        yield {
            'virtual_memory': mock_vmem,
            'swap_memory': mock_swap
        }


@pytest.fixture
def mock_psutil_disk():
    """Mock psutil disk functions with realistic data."""
    with patch('psutil.disk_partitions') as mock_partitions, \
         patch('psutil.disk_usage') as mock_usage:
        
        # Mock partition
        mock_partition = Mock()
        mock_partition.mountpoint = '/'
        mock_partition.fstype = 'ext4'
        mock_partitions.return_value = [mock_partition]
        
        # Mock disk usage
        mock_usage_obj = Mock()
        mock_usage_obj.total = 500 * 1024**3  # 500 GB
        mock_usage_obj.used = 250 * 1024**3   # 250 GB used
        mock_usage_obj.free = 250 * 1024**3   # 250 GB free
        mock_usage_obj.percent = 50.0
        mock_usage.return_value = mock_usage_obj
        
        yield {
            'disk_partitions': mock_partitions,
            'disk_usage': mock_usage
        }


@pytest.fixture
def mock_psutil_network():
    """Mock psutil network functions with realistic data."""
    with patch('psutil.net_io_counters') as mock_net_io:
        
        mock_stats = Mock()
        mock_stats.bytes_sent = 1024 * 1024 * 100  # 100 MB sent
        mock_stats.bytes_recv = 1024 * 1024 * 200  # 200 MB received
        mock_stats.packets_sent = 10000
        mock_stats.packets_recv = 15000
        mock_stats.errin = 0
        mock_stats.errout = 0
        mock_stats.dropin = 0
        mock_stats.dropout = 0
        
        mock_net_io.return_value = {'eth0': mock_stats}
        
        yield mock_net_io


@pytest.fixture
def mock_psutil_processes():
    """Mock psutil process functions with realistic data."""
    with patch('psutil.process_iter') as mock_proc_iter:
        
        # Create mock processes
        processes = []
        for i in range(5):
            mock_proc = Mock()
            mock_proc.info = {
                'pid': 1000 + i,
                'name': f'process_{i}',
                'username': 'testuser',
                'cpu_percent': 10.0 - i,  # Decreasing CPU usage
                'memory_percent': 5.0 + i  # Increasing memory usage
            }
            processes.append(mock_proc)
        
        mock_proc_iter.return_value = processes
        
        yield mock_proc_iter


@pytest.fixture
def mock_platform_info():
    """Mock platform information functions."""
    with patch('platform.node') as mock_node, \
         patch('platform.system') as mock_system, \
         patch('platform.release') as mock_release, \
         patch('platform.machine') as mock_machine, \
         patch('platform.processor') as mock_processor:
        
        mock_node.return_value = 'test-hostname'
        mock_system.return_value = 'Linux'
        mock_release.return_value = '5.4.0'
        mock_machine.return_value = 'x86_64'
        mock_processor.return_value = 'Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz'
        
        yield {
            'node': mock_node,
            'system': mock_system,
            'release': mock_release,
            'machine': mock_machine,
            'processor': mock_processor
        }


@pytest.fixture
def mock_datetime():
    """Mock datetime functions for consistent testing."""
    with patch('mcp_system_monitor_server.datetime') as mock_dt:
        from datetime import datetime
        fixed_time = datetime(2024, 1, 15, 12, 30, 45)
        mock_dt.now.return_value = fixed_time
        mock_dt.fromisoformat = datetime.fromisoformat
        yield mock_dt


# Test markers for organizing test runs
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU detection"
    )