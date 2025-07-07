from mcp.server.fastmcp import FastMCP, Context
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import psutil
import time
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPUtil has compatibility issues with Python 3.12+, using alternative approaches
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import nvidia_ml_py as nvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

import platform


class CPUInfo(BaseModel):
    usage_percent: float = Field(description="CPU usage percentage")
    usage_per_core: List[float] = Field(description="Per-core usage")
    frequency_current: float = Field(description="Current frequency in MHz")
    frequency_max: float = Field(description="Maximum frequency in MHz")
    core_count: int = Field(description="Number of CPU cores")
    thread_count: int = Field(description="Number of CPU threads")
    temperature: Optional[float] = Field(None, description="CPU temperature in Celsius")
    processor_name: str = Field(description="CPU model name")
    vendor: str = Field(description="CPU manufacturer")
    architecture: str = Field(description="CPU architecture")
    bits: int = Field(description="CPU bits (32 or 64)")
    cpu_family: Optional[str] = Field(None, description="CPU family")
    model: Optional[str] = Field(None, description="CPU model number")
    stepping: Optional[str] = Field(None, description="CPU stepping")
    cache_sizes: Optional[Dict[str, int]] = Field(None, description="CPU cache sizes in KB")


class GPUInfo(BaseModel):
    name: str = Field(description="GPU name")
    usage_percent: float = Field(description="GPU usage percentage")
    memory_used_mb: int = Field(description="Used VRAM in MB")
    memory_total_mb: int = Field(description="Total VRAM in MB")
    temperature: Optional[float] = Field(None, description="GPU temperature")
    power_usage: Optional[float] = Field(None, description="Power usage in watts")
    cores: Optional[int] = Field(None, description="Number of GPU cores (Apple Silicon)")
    unified_memory: Optional[bool] = Field(None, description="Whether GPU uses unified memory")
    metal_support: Optional[str] = Field(None, description="Metal API support version")


class MemoryInfo(BaseModel):
    total_gb: float = Field(description="Total RAM in GB")
    available_gb: float = Field(description="Available RAM in GB")
    used_gb: float = Field(description="Used RAM in GB")
    usage_percent: float = Field(description="Memory usage percentage")
    swap_total_gb: float = Field(description="Total swap in GB")
    swap_used_gb: float = Field(description="Used swap in GB")


class DiskInfo(BaseModel):
    mount_point: str = Field(description="Mount point or drive letter")
    total_gb: float = Field(description="Total disk space in GB")
    used_gb: float = Field(description="Used disk space in GB")
    free_gb: float = Field(description="Free disk space in GB")
    usage_percent: float = Field(description="Disk usage percentage")
    filesystem: str = Field(description="Filesystem type")


class SystemInfo(BaseModel):
    timestamp: datetime = Field(description="Data collection timestamp")
    hostname: str = Field(description="System hostname")
    platform: str = Field(description="Operating system")
    architecture: str = Field(description="System architecture")
    uptime_seconds: int = Field(description="System uptime in seconds")


# Phase 1: Performance Enhancement Models
class IOPerformanceInfo(BaseModel):
    """I/O Performance metrics"""
    read_bytes_per_sec: float = Field(description="Disk read bytes per second")
    write_bytes_per_sec: float = Field(description="Disk write bytes per second")
    read_count_per_sec: float = Field(description="Read operations per second")
    write_count_per_sec: float = Field(description="Write operations per second")
    read_time_ms: float = Field(description="Average read time in milliseconds")
    write_time_ms: float = Field(description="Average write time in milliseconds")
    busy_time_percent: float = Field(description="Disk busy time percentage")
    per_disk_stats: List[Dict[str, Any]] = Field(description="Per-disk I/O statistics")


class SystemLoadInfo(BaseModel):
    """System load and performance metrics"""
    load_average_1m: Optional[float] = Field(None, description="1-minute load average")
    load_average_5m: Optional[float] = Field(None, description="5-minute load average") 
    load_average_15m: Optional[float] = Field(None, description="15-minute load average")
    context_switches_per_sec: float = Field(description="Context switches per second")
    interrupts_per_sec: float = Field(description="Interrupts per second")
    processes_running: int = Field(description="Number of running processes")
    processes_blocked: int = Field(description="Number of blocked processes")
    boot_time: datetime = Field(description="System boot time")


class EnhancedMemoryInfo(BaseModel):
    """Enhanced memory statistics beyond basic usage"""
    total_gb: float = Field(description="Total RAM in GB")
    available_gb: float = Field(description="Available RAM in GB")
    used_gb: float = Field(description="Used RAM in GB")
    usage_percent: float = Field(description="Memory usage percentage")
    swap_total_gb: float = Field(description="Total swap in GB")
    swap_used_gb: float = Field(description="Used swap in GB")
    # Enhanced metrics
    buffers_gb: Optional[float] = Field(None, description="Buffer memory in GB")
    cached_gb: Optional[float] = Field(None, description="Cached memory in GB")
    shared_gb: Optional[float] = Field(None, description="Shared memory in GB")
    active_gb: Optional[float] = Field(None, description="Active memory in GB")
    inactive_gb: Optional[float] = Field(None, description="Inactive memory in GB")
    page_faults_per_sec: Optional[float] = Field(None, description="Page faults per second")
    swap_in_per_sec: Optional[float] = Field(None, description="Swap in operations per second")
    swap_out_per_sec: Optional[float] = Field(None, description="Swap out operations per second")


class EnhancedNetworkInfo(BaseModel):
    """Enhanced network statistics with performance metrics"""
    interface_name: str = Field(description="Network interface name")
    bytes_sent: int = Field(description="Total bytes sent")
    bytes_recv: int = Field(description="Total bytes received")
    packets_sent: int = Field(description="Total packets sent")
    packets_recv: int = Field(description="Total packets received")
    # Enhanced metrics
    bytes_sent_per_sec: float = Field(description="Bytes sent per second")
    bytes_recv_per_sec: float = Field(description="Bytes received per second") 
    packets_sent_per_sec: float = Field(description="Packets sent per second")
    packets_recv_per_sec: float = Field(description="Packets received per second")
    errors_in: int = Field(description="Input errors")
    errors_out: int = Field(description="Output errors")
    drops_in: int = Field(description="Input packets dropped")
    drops_out: int = Field(description="Output packets dropped")
    speed_mbps: Optional[float] = Field(None, description="Interface speed in Mbps")
    mtu: Optional[int] = Field(None, description="Maximum transmission unit")
    is_up: bool = Field(description="Interface is up")


class SystemPerformanceSnapshot(BaseModel):
    """Complete system performance snapshot"""
    io_performance: IOPerformanceInfo
    system_load: SystemLoadInfo
    enhanced_memory: EnhancedMemoryInfo
    enhanced_network: List[EnhancedNetworkInfo]
    collection_time: datetime = Field(default_factory=datetime.now)


class SystemSnapshot(BaseModel):
    """Complete system information snapshot"""
    system: SystemInfo
    cpu: CPUInfo
    gpus: List[GPUInfo]
    memory: MemoryInfo
    disks: List[DiskInfo]
    collection_time: datetime = Field(default_factory=datetime.now)


class BaseCollector(ABC):

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._cache: Dict[str, Any] = {}
        self._last_update: Optional[float] = None
        self._lock = asyncio.Lock()

    @abstractmethod
    async def collect_data(self) -> Dict[str, Any]:
        pass

    async def get_cached_data(self, max_age: float = 2.0) -> Dict[str, Any]:
        async with self._lock:
            current_time = time.time()

            if (self._last_update is None or
                    current_time - self._last_update > max_age):
                logger.debug(f"{self.__class__.__name__}: Collecting fresh data")
                try:
                    self._cache = await self.collect_data()
                    self._last_update = current_time
                except Exception as e:
                    logger.error(f"{self.__class__.__name__}: Error collecting data: {e}")
                    raise

            return self._cache


class CPUCollector(BaseCollector):
    """Collector for CPU information."""

    def __init__(self):
        super().__init__()
        self._static_cpu_info = self._get_static_cpu_info()

    def _get_static_cpu_info(self) -> Dict[str, Any]:
        """Get static CPU information that doesn't change during runtime."""
        import subprocess

        # Get basic processor info
        processor_name = platform.processor() or "Unknown"
        architecture = platform.machine()
        bits = 64 if platform.machine().endswith('64') else 32

        # Try to get more detailed CPU info
        vendor = "Unknown"
        cpu_family = None
        model = None
        stepping = None
        cache_sizes = {}

        # Platform-specific CPU info gathering
        if platform.system() == "Windows":
            try:
                # Use wmic to get CPU details
                result = subprocess.run(
                    ["wmic", "cpu", "get", "Name,Manufacturer,Family,Model,Stepping,L2CacheSize,L3CacheSize", "/value"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            if key == "Name" and value:
                                processor_name = value.strip()
                            elif key == "Manufacturer" and value:
                                vendor = value.strip()
                            elif key == "Family" and value:
                                cpu_family = value.strip()
                            elif key == "Model" and value:
                                model = value.strip()
                            elif key == "Stepping" and value:
                                stepping = value.strip()
                            elif key == "L2CacheSize" and value:
                                try:
                                    cache_sizes["L2"] = int(value.strip())
                                except ValueError:
                                    pass
                            elif key == "L3CacheSize" and value:
                                try:
                                    cache_sizes["L3"] = int(value.strip())
                                except ValueError:
                                    pass
            except Exception as e:
                logger.debug(f"Failed to get detailed CPU info via wmic: {e}")

        elif platform.system() == "Linux":
            try:
                # Parse /proc/cpuinfo for detailed information
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()

                for line in cpuinfo.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        if key == "model name" and value:
                            processor_name = value
                        elif key == "vendor_id" and value:
                            vendor = value
                        elif key == "cpu family" and value:
                            cpu_family = value
                        elif key == "model" and value and not model:
                            model = value
                        elif key == "stepping" and value:
                            stepping = value
                        elif key == "cache size" and value:
                            # Parse cache size (usually in KB)
                            try:
                                cache_kb = int(value.replace(' KB', '').strip())
                                cache_sizes["L3"] = cache_kb
                            except ValueError:
                                pass

                # Try lscpu for additional cache info
                result = subprocess.run(
                    ["lscpu"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'L1d cache:' in line:
                            try:
                                value = line.split(':')[1].strip()
                                if 'K' in value:
                                    cache_sizes["L1d"] = int(value.replace('K', '').strip())
                            except:
                                pass
                        elif 'L1i cache:' in line:
                            try:
                                value = line.split(':')[1].strip()
                                if 'K' in value:
                                    cache_sizes["L1i"] = int(value.replace('K', '').strip())
                            except:
                                pass
                        elif 'L2 cache:' in line:
                            try:
                                value = line.split(':')[1].strip()
                                if 'K' in value:
                                    cache_sizes["L2"] = int(value.replace('K', '').strip())
                                elif 'M' in value:
                                    cache_sizes["L2"] = int(value.replace('M', '').strip()) * 1024
                            except:
                                pass

            except Exception as e:
                logger.debug(f"Failed to parse /proc/cpuinfo: {e}")

        elif platform.system() == "Darwin":  # macOS
            try:
                # Use sysctl to get CPU info
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    processor_name = result.stdout.strip()

                # Get vendor
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.vendor"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    vendor = result.stdout.strip()

                # Get cache sizes
                for cache_level, sysctl_name in [
                    ("L1i", "hw.l1icachesize"),
                    ("L1d", "hw.l1dcachesize"),
                    ("L2", "hw.l2cachesize"),
                    ("L3", "hw.l3cachesize")
                ]:
                    try:
                        result = subprocess.run(
                            ["sysctl", "-n", sysctl_name],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            cache_bytes = int(result.stdout.strip())
                            if cache_bytes > 0:
                                cache_sizes[cache_level] = cache_bytes // 1024  # Convert to KB
                    except:
                        pass

            except Exception as e:
                logger.debug(f"Failed to get CPU info via sysctl: {e}")

        # Clean up processor name and extract vendor if needed
        if processor_name and vendor == "Unknown":
            processor_lower = processor_name.lower()
            if "intel" in processor_lower:
                vendor = "Intel"
            elif "amd" in processor_lower:
                vendor = "AMD"
            elif "apple" in processor_lower:
                vendor = "Apple"
            elif "arm" in processor_lower:
                vendor = "ARM"

        return {
            "processor_name": processor_name,
            "vendor": vendor,
            "architecture": architecture,
            "bits": bits,
            "cpu_family": cpu_family,
            "model": model,
            "stepping": stepping,
            "cache_sizes": cache_sizes if cache_sizes else None
        }

    async def collect_data(self) -> Dict[str, Any]:
        """Collects CPU data."""
        usage_percent = psutil.cpu_percent(interval=1)
        usage_per_core = psutil.cpu_percent(interval=1, percpu=True)

        freq = psutil.cpu_freq()
        frequency_current = freq.current if freq else 0
        frequency_max = freq.max if freq else 0

        core_count = psutil.cpu_count(logical=False)
        thread_count = psutil.cpu_count(logical=True)

        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temperature = temps['coretemp'][0].current
        except (AttributeError, KeyError):
            # sensors_temperatures not available on all platforms
            logger.debug("CPU temperature sensors not available on this platform")
        except Exception as e:
            logger.warning(f"Unexpected error reading CPU temperature: {e}")

        # Combine dynamic and static data
        return {
            "usage_percent": usage_percent,
            "usage_per_core": usage_per_core,
            "frequency_current": frequency_current,
            "frequency_max": frequency_max,
            "core_count": core_count,
            "thread_count": thread_count,
            "temperature": temperature,
            **self._static_cpu_info
        }


class DiskCollector(BaseCollector):
    """Collector for disk information."""

    async def collect_data(self) -> Dict[str, Any]:
        """Collects disk data."""
        disks_info = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks_info.append({
                    "mount_point": str(partition.mountpoint).replace("\\", "\\\\"),
                    "total_gb": usage.total / (1024 ** 3),
                    "used_gb": usage.used / (1024 ** 3),
                    "free_gb": usage.free / (1024 ** 3),
                    "usage_percent": usage.percent,
                    "filesystem": partition.fstype,
                })
            except (PermissionError, FileNotFoundError) as e:
                # Ignore drives that are not ready, e.g. CD-ROMs
                logger.debug(f"Skipping inaccessible drive {partition.mountpoint}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error accessing drive {partition.mountpoint}: {e}")
                continue
        return {"disks": disks_info}


class GPUCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("GPU monitoring initialized via pynvml")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML via pynvml: {e}")
        elif NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
                logger.info("GPU monitoring initialized via nvidia-ml-py")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML via nvidia-ml-py: {e}")
        else:
            logger.info("No NVIDIA GPU monitoring library available, will try generic methods")

    def _get_generic_gpu_info_windows(self) -> List[Dict[str, Any]]:
        """Get GPU info on Windows using WMI via subprocess."""
        gpus = []
        try:
            import subprocess

            # Get GPU info via wmic
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,CurrentRefreshRate,VideoProcessor",
                 "/value"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                current_gpu = {}
                gpu_id = 0

                for line in result.stdout.strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip()

                        if key == "Name" and value:
                            if current_gpu:
                                # Process previous GPU
                                gpus.append(current_gpu)
                                gpu_id += 1
                            current_gpu = {
                                "id": gpu_id,
                                "name": value,
                                "usage_percent": 0.0,  # Not available via WMI
                                "memory_used_mb": 0,
                                "memory_total_mb": 0,
                                "temperature": None,
                                "power_usage": None
                            }
                        elif key == "AdapterRAM" and value and current_gpu:
                            try:
                                # AdapterRAM is in bytes
                                memory_bytes = int(value)
                                if memory_bytes > 0:
                                    current_gpu["memory_total_mb"] = memory_bytes / 1024 / 1024  # Convert to MB
                            except ValueError:
                                pass
                        elif key == "VideoProcessor" and value and current_gpu:
                            # Add video processor info to name if available
                            if value and value != current_gpu.get("name", ""):
                                current_gpu["name"] = f"{current_gpu['name']} ({value})"

                # Don't forget the last GPU
                if current_gpu and "name" in current_gpu:
                    gpus.append(current_gpu)

            # Try to get GPU usage via performance counters
            if gpus:
                try:
                    # Get GPU utilization
                    result = subprocess.run(
                        ["wmic", "path", "Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine",
                         "where", "Name like '%engtype_3D%'", "get", "UtilizationPercentage", "/value"],
                        capture_output=True, text=True, timeout=5
                    )

                    if result.returncode == 0:
                        utilizations = []
                        for line in result.stdout.strip().split('\n'):
                            if 'UtilizationPercentage=' in line:
                                try:
                                    util = float(line.split('=')[1].strip())
                                    utilizations.append(util)
                                except:
                                    pass

                        # Apply utilization to first GPU (simplified approach)
                        if utilizations and gpus:
                            gpus[0]["usage_percent"] = max(utilizations)
                except:
                    pass

        except Exception as e:
            logger.debug(f"Failed to get GPU info via WMI: {e}")

        return gpus

    def _get_generic_gpu_info_linux(self) -> List[Dict[str, Any]]:
        """Get GPU info on Linux using various methods."""
        gpus = []

        # Try lspci first
        try:
            import subprocess

            result = subprocess.run(
                ["lspci", "-v", "-nn"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                current_gpu = None
                gpu_id = 0

                for line in result.stdout.split('\n'):
                    # Look for VGA or 3D controller
                    if 'VGA compatible controller' in line or '3D controller' in line:
                        # Extract GPU name
                        parts = line.split(': ', 1)
                        if len(parts) > 1:
                            name = parts[1].split(' [')[0].strip()
                            current_gpu = {
                                "id": gpu_id,
                                "name": name,
                                "usage_percent": 0.0,
                                "memory_used_mb": 0,
                                "memory_total_mb": 0,
                                "temperature": None,
                                "power_usage": None
                            }
                            gpu_id += 1
                    elif current_gpu and 'Memory at' in line and 'size=' in line:
                        # Try to extract memory size
                        try:
                            size_part = line.split('size=')[1].split(']')[0]
                            if 'M' in size_part:
                                size_mb = int(size_part.replace('M', ''))
                                current_gpu["memory_total_mb"] = size_mb
                            elif 'G' in size_part:
                                size_gb = int(size_part.replace('G', ''))
                                current_gpu["memory_total_mb"] = size_gb * 1024
                        except:
                            pass
                    elif current_gpu and line.strip() == '':
                        # Empty line indicates end of device info
                        gpus.append(current_gpu)
                        current_gpu = None

                # Don't forget the last GPU
                if current_gpu:
                    gpus.append(current_gpu)

        except Exception as e:
            logger.debug(f"Failed to get GPU info via lspci: {e}")

        # Try to get Intel GPU info
        try:
            import subprocess

            # Check for Intel GPU
            result = subprocess.run(
                ["cat", "/sys/class/drm/card0/device/vendor"],
                capture_output=True, text=True, timeout=2
            )

            if result.returncode == 0 and "0x8086" in result.stdout:  # Intel vendor ID
                # This is an Intel GPU
                intel_gpu = {
                    "id": len(gpus),
                    "name": "Intel Integrated Graphics",
                    "usage_percent": 0.0,
                    "memory_used_mb": 0,
                    "memory_total_mb": 0,
                    "temperature": None,
                    "power_usage": None
                }

                # Try to get more specific name
                try:
                    result = subprocess.run(
                        ["cat", "/sys/class/drm/card0/device/label"],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        intel_gpu["name"] = f"Intel {result.stdout.strip()}"
                except:
                    pass

                # Try to get temperature
                try:
                    result = subprocess.run(
                        ["cat", "/sys/class/drm/card0/gt/gt0/throttle_reason_status"],
                        capture_output=True, text=True, timeout=2
                    )
                    # This is a simplified approach - actual temperature reading would be more complex
                except:
                    pass

                # Only add if we haven't already detected this GPU
                if not any(gpu for gpu in gpus if "Intel" in gpu.get("name", "")):
                    gpus.append(intel_gpu)

        except:
            pass

        # Try AMD GPU detection
        try:
            import subprocess
            import os

            # Check for AMD GPUs in /sys/class/drm/
            for card in os.listdir("/sys/class/drm/"):
                if card.startswith("card") and card[4:].isdigit():
                    vendor_path = f"/sys/class/drm/{card}/device/vendor"
                    if os.path.exists(vendor_path):
                        with open(vendor_path, 'r') as f:
                            vendor = f.read().strip()

                        if "0x1002" in vendor:  # AMD vendor ID
                            amd_gpu = {
                                "id": len(gpus),
                                "name": "AMD Graphics",
                                "usage_percent": 0.0,
                                "memory_used_mb": 0,
                                "memory_total_mb": 0,
                                "temperature": None,
                                "power_usage": None
                            }

                            # Try to get model name
                            try:
                                result = subprocess.run(
                                    ["cat", f"/sys/class/drm/{card}/device/product_name"],
                                    capture_output=True, text=True, timeout=2
                                )
                                if result.returncode == 0 and result.stdout.strip():
                                    amd_gpu["name"] = result.stdout.strip()
                            except:
                                pass

                            # Try to get temperature
                            try:
                                temp_path = f"/sys/class/drm/{card}/device/hwmon/hwmon*/temp1_input"
                                import glob
                                temp_files = glob.glob(temp_path)
                                if temp_files:
                                    with open(temp_files[0], 'r') as f:
                                        temp_milli = int(f.read().strip())
                                        amd_gpu["temperature"] = temp_milli / 1000.0
                            except:
                                pass

                            # Only add if we haven't already detected this GPU
                            if not any(gpu for gpu in gpus if amd_gpu["name"] in gpu.get("name", "")):
                                gpus.append(amd_gpu)

        except Exception as e:
            logger.debug(f"Failed to get AMD GPU info: {e}")

        return gpus

    def _get_generic_gpu_info_macos(self) -> List[Dict[str, Any]]:
        """Get GPU info on macOS using system_profiler."""
        gpus = []
        try:
            import subprocess
            import json

            # Use system_profiler to get GPU info
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                displays_data = data.get("SPDisplaysDataType", [])

                gpu_id = 0
                for display in displays_data:
                    # Check if this is actually a GPU device
                    device_type = display.get("sppci_device_type")
                    if device_type != "spdisplays_gpu":
                        continue

                    # Get GPU name
                    gpu_name = display.get("sppci_model", display.get("_name", "Unknown GPU"))

                    gpu_info = {
                        "id": gpu_id,
                        "name": gpu_name,
                        "usage_percent": 0.0,  # Changed from "load" to match GPUInfo model
                        "memory_used_mb": 0,   # Changed from "memory_used" to match GPUInfo model
                        "memory_total_mb": 0,  # Changed from "memory_total" to match GPUInfo model
                        "temperature": None,
                        "power_usage": None
                    }

                    # Get GPU core count if available (Apple Silicon specific)
                    cores = display.get("sppci_cores")
                    if cores:
                        try:
                            gpu_info["cores"] = int(cores)
                        except (ValueError, TypeError):
                            pass

                    # Try to get VRAM info (traditional discrete GPUs)
                    vram = display.get("spdisplays_vram") or display.get("sppci_vram")
                    if vram:
                        # Parse VRAM string (e.g., "1536 MB", "8 GB")
                        try:
                            parts = vram.split()
                            if len(parts) >= 2:
                                value = float(parts[0])
                                unit = parts[1].upper()
                                if "GB" in unit:
                                    gpu_info["memory_total_mb"] = value * 1024  # Convert to MB
                                elif "MB" in unit:
                                    gpu_info["memory_total_mb"] = value
                        except (ValueError, IndexError):
                            pass
                    else:
                        # For Apple Silicon GPUs with unified memory, try to get system memory
                        # as they share memory with the CPU
                        try:
                            import psutil
                            total_memory = psutil.virtual_memory().total
                            # Apple Silicon GPUs can access most of system memory
                            # But we'll report a conservative estimate
                            if "Apple" in gpu_name and ("M1" in gpu_name or "M2" in gpu_name or "M3" in gpu_name):
                                # Unified memory - report total system memory in MB
                                gpu_info["memory_total_mb"] = total_memory / (1024 * 1024)
                                gpu_info["unified_memory"] = True
                        except Exception:
                            pass

                    # Check for Metal support (indicates it's a capable GPU)
                    metal_support = display.get("spdisplays_mtlgpufamilysupport") or display.get("spdisplays_metal")
                    has_metal = metal_support is not None

                    # Additional check: ensure it has connected displays or is a discrete GPU
                    has_displays = bool(display.get("spdisplays_ndrvs"))
                    
                    # Apple Silicon GPUs always have Metal support and should be included
                    is_apple_silicon = "Apple" in gpu_name and any(chip in gpu_name for chip in ["M1", "M2", "M3"])
                    
                    if has_metal or has_displays or is_apple_silicon:
                        # Add Metal version info if available
                        if metal_support:
                            gpu_info["metal_support"] = metal_support
                        
                        gpus.append(gpu_info)
                        gpu_id += 1

        except Exception as e:
            logger.debug(f"Failed to get GPU info via system_profiler: {e}")

        # Try to get additional GPU temperature info via powermetrics (requires sudo)
        if gpus:
            try:
                import subprocess
                
                # Try powermetrics for Apple Silicon GPU metrics (if available without sudo)
                result = subprocess.run(
                    ["powermetrics", "--samplers", "gpu_power", "-n", "1", "-f", "plist"],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0:
                    # Parse plist output for GPU power/thermal data
                    # This is complex and would require plist parsing
                    # For now, we'll skip this advanced feature
                    pass
                    
            except Exception as e:
                logger.debug(f"Failed to get additional GPU metrics via powermetrics: {e}")

        return gpus

    async def collect_data(self) -> Dict[str, Any]:
        gpus = []

        # First try NVIDIA GPUs with existing code
        if self.nvml_initialized and PYNVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')

                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None

                    # Get power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power = None

                    gpus.append({
                        "id": i,
                        "name": name,
                        "usage_percent": float(util.gpu),
                        "memory_used_mb": int(mem_info.used / 1024 / 1024),  # Convert to MB
                        "memory_total_mb": int(mem_info.total / 1024 / 1024),  # Convert to MB
                        "temperature": temp,
                        "power_usage": power
                    })
            except Exception as e:
                logger.error(f"Error collecting GPU data with pynvml: {e}")

        elif self.nvml_initialized and NVML_AVAILABLE:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')

                    # Get memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                    # Get utilization
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)

                    # Get temperature
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = None

                    gpus.append({
                        "id": i,
                        "name": name,
                        "usage_percent": float(util.gpu),
                        "memory_used_mb": int(mem_info.used / 1024 / 1024),  # Convert to MB
                        "memory_total_mb": int(mem_info.total / 1024 / 1024),  # Convert to MB
                        "temperature": temp,
                        "power_usage": None
                    })
            except Exception as e:
                logger.error(f"Error collecting GPU data with nvidia-ml-py: {e}")

        # If no NVIDIA GPUs found or NVML not available, try generic methods
        if not gpus:
            system = platform.system()

            if system == "Windows":
                gpus = self._get_generic_gpu_info_windows()
            elif system == "Linux":
                gpus = self._get_generic_gpu_info_linux()
            elif system == "Darwin":
                gpus = self._get_generic_gpu_info_macos()
            else:
                logger.info(f"Generic GPU detection not implemented for {system}")

        return {"gpus": gpus}

    def __del__(self):
        if self.nvml_initialized:
            try:
                if PYNVML_AVAILABLE:
                    pynvml.nvmlShutdown()
                elif NVML_AVAILABLE:
                    nvml.nvmlShutdown()
            except:
                pass


class MemoryCollector(BaseCollector):
    """Collector for memory information."""

    async def collect_data(self) -> Dict[str, Any]:
        """Collects memory data."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "total_gb": mem.total / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "used_gb": mem.used / (1024 ** 3),
            "usage_percent": mem.percent,
            "swap_total_gb": swap.total / (1024 ** 3),
            "swap_used_gb": swap.used / (1024 ** 3),
        }


class NetworkCollector(BaseCollector):
    """Collector for network information."""

    async def collect_data(self) -> Dict[str, Any]:
        """Collects network data."""
        net_io = psutil.net_io_counters(pernic=True)
        return {"interfaces": {if_name: {
            "bytes_sent": stats.bytes_sent,
            "bytes_recv": stats.bytes_recv,
            "packets_sent": stats.packets_sent,
            "packets_recv": stats.packets_recv,
            "errin": stats.errin,
            "errout": stats.errout,
            "dropin": stats.dropin,
            "dropout": stats.dropout,
        } for if_name, stats in net_io.items()}}


class ProcessCollector(BaseCollector):
    """Collector for process information."""

    async def collect_data(self) -> Dict[str, Any]:
        """Collects process data."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logger.debug(f"Skipping process {proc.info.get('pid', 'unknown')}: {type(e).__name__}")
            except Exception as e:
                logger.warning(f"Unexpected error accessing process {proc.info.get('pid', 'unknown')}: {e}")
        return {"processes": processes}

    async def get_top_processes(self, limit: int = 10, sort_by: str = 'cpu_percent') -> List[Dict[str, Any]]:
        """Get top processes sorted by a given metric."""
        data = await self.get_cached_data()
        processes = data.get("processes", [])

        # Filter out processes with None values for the sort key
        processes = [p for p in processes if p.get(sort_by) is not None]

        sorted_processes = sorted(processes, key=lambda p: p.get(sort_by, 0), reverse=True)
        return sorted_processes[:limit]


class SystemCollector(BaseCollector):
    """Collector for general system information."""

    async def collect_data(self) -> Dict[str, Any]:
        """Collects general system data."""
        boot_time = psutil.boot_time()
        uptime_seconds = int(datetime.now().timestamp() - boot_time)

        return {
            "timestamp": datetime.now(),
            "hostname": platform.node(),
            "platform": f"{platform.system()} {platform.release()}",
            "architecture": platform.machine(),
            "uptime_seconds": uptime_seconds,
        }


# Phase 1: Performance Enhancement Collectors
class IOPerformanceCollector(BaseCollector):
    """Collector for I/O performance metrics"""
    
    def __init__(self):
        super().__init__()
        self._last_io_stats = None
        self._last_io_time = None
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect I/O performance data"""
        current_time = time.time()
        current_stats = psutil.disk_io_counters()
        
        # Initialize per-disk stats
        per_disk_stats = []
        try:
            per_disk = psutil.disk_io_counters(perdisk=True)
            for disk_name, disk_stats in per_disk.items():
                per_disk_stats.append({
                    "device": disk_name,
                    "read_bytes": disk_stats.read_bytes,
                    "write_bytes": disk_stats.write_bytes,
                    "read_count": disk_stats.read_count,
                    "write_count": disk_stats.write_count,
                    "read_time_ms": disk_stats.read_time,
                    "write_time_ms": disk_stats.write_time,
                    "busy_time_ms": getattr(disk_stats, 'busy_time', 0)
                })
        except Exception as e:
            logger.debug(f"Could not get per-disk stats: {e}")
        
        # Calculate rates if we have previous data
        if self._last_io_stats and self._last_io_time:
            time_delta = current_time - self._last_io_time
            if time_delta > 0:
                read_bytes_per_sec = (current_stats.read_bytes - self._last_io_stats.read_bytes) / time_delta
                write_bytes_per_sec = (current_stats.write_bytes - self._last_io_stats.write_bytes) / time_delta
                read_count_per_sec = (current_stats.read_count - self._last_io_stats.read_count) / time_delta
                write_count_per_sec = (current_stats.write_count - self._last_io_stats.write_count) / time_delta
            else:
                read_bytes_per_sec = write_bytes_per_sec = 0
                read_count_per_sec = write_count_per_sec = 0
        else:
            read_bytes_per_sec = write_bytes_per_sec = 0
            read_count_per_sec = write_count_per_sec = 0
        
        # Calculate average I/O times
        read_time_ms = current_stats.read_time / max(current_stats.read_count, 1)
        write_time_ms = current_stats.write_time / max(current_stats.write_count, 1)
        
        # Calculate busy time percentage (approximation)
        total_io_time = current_stats.read_time + current_stats.write_time
        busy_time_percent = min(100.0, (total_io_time / (current_time * 1000)) * 100) if current_time > 0 else 0
        
        # Store for next calculation
        self._last_io_stats = current_stats
        self._last_io_time = current_time
        
        return {
            "read_bytes_per_sec": read_bytes_per_sec,
            "write_bytes_per_sec": write_bytes_per_sec,
            "read_count_per_sec": read_count_per_sec,
            "write_count_per_sec": write_count_per_sec,
            "read_time_ms": read_time_ms,
            "write_time_ms": write_time_ms,
            "busy_time_percent": busy_time_percent,
            "per_disk_stats": per_disk_stats
        }


class SystemLoadCollector(BaseCollector):
    """Collector for system load and performance metrics"""
    
    def __init__(self):
        super().__init__()
        self._last_cpu_stats = None
        self._last_stats_time = None
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect system load data"""
        current_time = time.time()
        
        # Get load averages (Unix-like systems only)
        load_averages = None
        try:
            if hasattr(psutil, 'getloadavg'):
                load_1, load_5, load_15 = psutil.getloadavg()
                load_averages = {
                    "load_average_1m": load_1,
                    "load_average_5m": load_5,
                    "load_average_15m": load_15
                }
        except (AttributeError, OSError):
            # Windows doesn't have load averages
            load_averages = {
                "load_average_1m": None,
                "load_average_5m": None,
                "load_average_15m": None
            }
        
        # Get CPU times for context switches and interrupts calculation
        cpu_times = psutil.cpu_times()
        
        # Calculate context switches and interrupts per second
        context_switches_per_sec = 0
        interrupts_per_sec = 0
        
        if self._last_cpu_stats and self._last_stats_time:
            time_delta = current_time - self._last_stats_time
            if time_delta > 0:
                try:
                    # Some platforms provide these stats
                    current_ctx_switches = getattr(cpu_times, 'ctx_switches', 0)
                    current_interrupts = getattr(cpu_times, 'interrupts', 0)
                    
                    last_ctx_switches = getattr(self._last_cpu_stats, 'ctx_switches', 0)
                    last_interrupts = getattr(self._last_cpu_stats, 'interrupts', 0)
                    
                    context_switches_per_sec = (current_ctx_switches - last_ctx_switches) / time_delta
                    interrupts_per_sec = (current_interrupts - last_interrupts) / time_delta
                except AttributeError:
                    # Platform doesn't provide these stats
                    pass
        
        self._last_cpu_stats = cpu_times
        self._last_stats_time = current_time
        
        # Count running and blocked processes
        processes_running = 0
        processes_blocked = 0
        
        try:
            for proc in psutil.process_iter(['status']):
                try:
                    status = proc.info['status']
                    if status in [psutil.STATUS_RUNNING, psutil.STATUS_WAKING]:
                        processes_running += 1
                    elif status in [psutil.STATUS_DISK_SLEEP, psutil.STATUS_STOPPED]:
                        processes_blocked += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.debug(f"Could not count process states: {e}")
        
        # Get boot time
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        result = {
            "context_switches_per_sec": context_switches_per_sec,
            "interrupts_per_sec": interrupts_per_sec,
            "processes_running": processes_running,
            "processes_blocked": processes_blocked,
            "boot_time": boot_time
        }
        result.update(load_averages)
        
        return result


class EnhancedMemoryCollector(BaseCollector):
    """Collector for enhanced memory statistics"""
    
    def __init__(self):
        super().__init__()
        self._last_memory_stats = None
        self._last_memory_time = None
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect enhanced memory data"""
        current_time = time.time()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Basic memory info
        result = {
            "total_gb": mem.total / (1024 ** 3),
            "available_gb": mem.available / (1024 ** 3),
            "used_gb": mem.used / (1024 ** 3),
            "usage_percent": mem.percent,
            "swap_total_gb": swap.total / (1024 ** 3),
            "swap_used_gb": swap.used / (1024 ** 3)
        }
        
        # Enhanced memory statistics (platform-dependent)
        try:
            # Linux-specific memory details
            if hasattr(mem, 'buffers'):
                result["buffers_gb"] = mem.buffers / (1024 ** 3)
            if hasattr(mem, 'cached'):
                result["cached_gb"] = mem.cached / (1024 ** 3)
            if hasattr(mem, 'shared'):
                result["shared_gb"] = mem.shared / (1024 ** 3)
            if hasattr(mem, 'active'):
                result["active_gb"] = mem.active / (1024 ** 3)
            if hasattr(mem, 'inactive'):
                result["inactive_gb"] = mem.inactive / (1024 ** 3)
        except AttributeError:
            # Platform doesn't provide these stats
            pass
        
        # Calculate page faults and swap rates per second
        page_faults_per_sec = None
        swap_in_per_sec = None
        swap_out_per_sec = None
        
        try:
            # Try to get memory statistics (Linux)
            import os
            if os.path.exists('/proc/vmstat'):
                with open('/proc/vmstat', 'r') as f:
                    vmstat = f.read()
                    
                current_stats = {}
                for line in vmstat.split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            current_stats[parts[0]] = int(parts[1])
                
                if self._last_memory_stats and self._last_memory_time:
                    time_delta = current_time - self._last_memory_time
                    if time_delta > 0:
                        # Calculate rates
                        pgfault_current = current_stats.get('pgfault', 0)
                        pgfault_last = self._last_memory_stats.get('pgfault', 0)
                        page_faults_per_sec = (pgfault_current - pgfault_last) / time_delta
                        
                        pswpin_current = current_stats.get('pswpin', 0)
                        pswpin_last = self._last_memory_stats.get('pswpin', 0)
                        swap_in_per_sec = (pswpin_current - pswpin_last) / time_delta
                        
                        pswpout_current = current_stats.get('pswpout', 0)
                        pswpout_last = self._last_memory_stats.get('pswpout', 0)
                        swap_out_per_sec = (pswpout_current - pswpout_last) / time_delta
                
                self._last_memory_stats = current_stats
                self._last_memory_time = current_time
                
        except Exception as e:
            logger.debug(f"Could not get enhanced memory stats: {e}")
        
        if page_faults_per_sec is not None:
            result["page_faults_per_sec"] = page_faults_per_sec
        if swap_in_per_sec is not None:
            result["swap_in_per_sec"] = swap_in_per_sec
        if swap_out_per_sec is not None:
            result["swap_out_per_sec"] = swap_out_per_sec
        
        return result


class EnhancedNetworkCollector(BaseCollector):
    """Collector for enhanced network statistics"""
    
    def __init__(self):
        super().__init__()
        self._last_network_stats = None
        self._last_network_time = None
    
    async def collect_data(self) -> Dict[str, Any]:
        """Collect enhanced network data"""
        current_time = time.time()
        net_io = psutil.net_io_counters(pernic=True)
        
        enhanced_interfaces = []
        
        for interface_name, stats in net_io.items():
            # Calculate rates if we have previous data
            bytes_sent_per_sec = 0
            bytes_recv_per_sec = 0
            packets_sent_per_sec = 0
            packets_recv_per_sec = 0
            
            if self._last_network_stats and interface_name in self._last_network_stats and self._last_network_time:
                time_delta = current_time - self._last_network_time
                if time_delta > 0:
                    last_stats = self._last_network_stats[interface_name]
                    bytes_sent_per_sec = (stats.bytes_sent - last_stats.bytes_sent) / time_delta
                    bytes_recv_per_sec = (stats.bytes_recv - last_stats.bytes_recv) / time_delta
                    packets_sent_per_sec = (stats.packets_sent - last_stats.packets_sent) / time_delta
                    packets_recv_per_sec = (stats.packets_recv - last_stats.packets_recv) / time_delta
            
            # Get interface details
            speed_mbps = None
            mtu = None
            is_up = False
            
            try:
                # Get network interface addresses and stats
                net_if_addrs = psutil.net_if_addrs()
                net_if_stats = psutil.net_if_stats()
                
                if interface_name in net_if_stats:
                    if_stats = net_if_stats[interface_name]
                    is_up = if_stats.isup
                    speed_mbps = if_stats.speed if if_stats.speed > 0 else None
                    mtu = if_stats.mtu
            except Exception as e:
                logger.debug(f"Could not get interface details for {interface_name}: {e}")
            
            enhanced_interfaces.append({
                "interface_name": interface_name,
                "bytes_sent": stats.bytes_sent,
                "bytes_recv": stats.bytes_recv,
                "packets_sent": stats.packets_sent,
                "packets_recv": stats.packets_recv,
                "bytes_sent_per_sec": bytes_sent_per_sec,
                "bytes_recv_per_sec": bytes_recv_per_sec,
                "packets_sent_per_sec": packets_sent_per_sec,
                "packets_recv_per_sec": packets_recv_per_sec,
                "errors_in": stats.errin,
                "errors_out": stats.errout,
                "drops_in": stats.dropin,
                "drops_out": stats.dropout,
                "speed_mbps": speed_mbps,
                "mtu": mtu,
                "is_up": is_up
            })
        
        # Store for next calculation
        self._last_network_stats = net_io
        self._last_network_time = current_time
        
        return {"enhanced_interfaces": enhanced_interfaces}


mcp = FastMCP(
    name="SystemMonitor",
    dependencies=[
        "psutil",
        "nvidia-ml-py",
        "pydantic",
        "pynvml"
    ]
)

# Globale Collector-Instanzen
logger.info("Initializing system collectors...")
cpu_collector = CPUCollector()
gpu_collector = GPUCollector()
memory_collector = MemoryCollector()
disk_collector = DiskCollector()
system_collector = SystemCollector()
process_collector = ProcessCollector()
network_collector = NetworkCollector()

# Phase 1: Performance Enhancement Collectors
io_performance_collector = IOPerformanceCollector()
system_load_collector = SystemLoadCollector()
enhanced_memory_collector = EnhancedMemoryCollector()
enhanced_network_collector = EnhancedNetworkCollector()

logger.info("System collectors initialized successfully")


@mcp.tool()
async def get_current_datetime() -> str:
    """Get the current local datetime in ISO Format YYYY-MM-DD HH:MM:SS"""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


@mcp.tool()
async def get_cpu_info() -> CPUInfo:
    """Get current CPU information including usage, frequency, and temperature.

    Returns comprehensive CPU metrics including:
    - Overall CPU usage percentage (0-100)
    - Per-core usage percentages
    - Current and maximum CPU frequency in MHz
    - Number of physical cores and logical threads
    - CPU temperature in Celsius (if available)
    - Processor name and model (e.g., "Intel Core i7-9750H")
    - CPU manufacturer (Intel, AMD, Apple, ARM, etc.)
    - Architecture (x86_64, ARM64, etc.)
    - CPU family, model, and stepping information
    - Cache sizes (L1, L2, L3) in KB

    Use this to monitor CPU performance, detect high load conditions,
    or gather detailed system specifications."""
    data = await cpu_collector.get_cached_data()
    return CPUInfo(**data)


@mcp.tool()
async def get_gpu_info() -> List[GPUInfo]:
    """Get information for all detected GPUs in the system.

    Returns a list of GPU information including:
    - GPU name and model (NVIDIA, AMD, Intel, Apple, etc.)
    - Current GPU utilization percentage (0-100) when available
    - VRAM usage in MB (used and total) when available
    - GPU temperature in Celsius when available
    - Power usage in watts (NVIDIA only)

    Supports:
    - NVIDIA GPUs: Full metrics via NVML
    - AMD GPUs: Basic detection and temperature (Linux)
    - Intel GPUs: Basic detection
    - Apple GPUs: Basic detection and VRAM info
    - Other GPUs: Basic detection via system tools

    Note: Some metrics may be None for non-NVIDIA GPUs.
    Use this to monitor GPU performance for ML workloads, gaming,
    or video processing tasks."""
    data = await gpu_collector.get_cached_data()
    gpu_list = []
    for gpu in data.get('gpus', []):
        gpu_list.append(GPUInfo(
            name=gpu.get('name', 'Unknown'),
            usage_percent=gpu.get('usage_percent', 0.0),
            memory_used_mb=gpu.get('memory_used_mb', 0),
            memory_total_mb=gpu.get('memory_total_mb', 0),
            temperature=gpu.get('temperature'),
            power_usage=gpu.get('power_usage'),
            cores=gpu.get('cores'),
            unified_memory=gpu.get('unified_memory'),
            metal_support=gpu.get('metal_support')
        ))
    return gpu_list


@mcp.tool()
async def get_memory_info() -> MemoryInfo:
    """Get current memory (RAM) usage information"""
    data = await memory_collector.get_cached_data()
    return MemoryInfo(**data)


@mcp.tool()
async def get_disk_info() -> List[DiskInfo]:
    """Get disk usage information for all mounted drives"""
    data = await disk_collector.get_cached_data()
    return [DiskInfo(**disk) for disk in data.get('disks', [])]


@mcp.tool()
async def get_system_snapshot() -> SystemSnapshot:
    """Get complete system information snapshot"""
    logger.debug("Collecting system snapshot")
    try:
        # Gather all data in parallel
        cpu_data, gpu_data, memory_data, disk_data, system_data = await asyncio.gather(
            cpu_collector.get_cached_data(),
            gpu_collector.get_cached_data(),
            memory_collector.get_cached_data(),
            disk_collector.get_cached_data(),
            system_collector.get_cached_data()
        )

        # Convert GPU data
        gpu_list = []
        for gpu in gpu_data.get('gpus', []):
            gpu_list.append(GPUInfo(
                name=gpu.get('name', 'Unknown'),
                usage_percent=gpu.get('usage_percent', 0.0),
                memory_used_mb=gpu.get('memory_used_mb', 0),
                memory_total_mb=gpu.get('memory_total_mb', 0),
                temperature=gpu.get('temperature'),
                power_usage=gpu.get('power_usage'),
                cores=gpu.get('cores'),
                unified_memory=gpu.get('unified_memory'),
                metal_support=gpu.get('metal_support')
            ))

        snapshot = SystemSnapshot(
            system=SystemInfo(**system_data),
            cpu=CPUInfo(**cpu_data),
            gpus=gpu_list,
            memory=MemoryInfo(**memory_data),
            disks=[DiskInfo(**disk) for disk in disk_data.get('disks', [])]
        )
        logger.info("System snapshot collected successfully")
        return snapshot
    except Exception as e:
        logger.error(f"Failed to collect system snapshot: {e}")
        raise


# Tools für Monitoring
@mcp.tool()
async def monitor_cpu_usage(duration_seconds: int = 5) -> Dict[str, Any]:
    """Monitor CPU usage over a specified duration"""
    logger.info(f"Starting CPU monitoring for {duration_seconds} seconds")
    samples = []
    for i in range(duration_seconds):
        data = await cpu_collector.collect_data()
        samples.append(data['usage_percent'])
        logger.debug(f"CPU sample {i + 1}/{duration_seconds}: {data['usage_percent']:.1f}%")
        await asyncio.sleep(1)

    result = {
        "average": sum(samples) / len(samples) if samples else 0,
        "minimum": min(samples) if samples else 0,
        "maximum": max(samples) if samples else 0,
        "samples": samples
    }
    logger.info(f"CPU monitoring complete. Average: {result['average']:.1f}%")
    return result


@mcp.tool()
async def get_top_processes(limit: int = 10, sort_by: str = 'cpu_percent') -> List[Dict[str, Any]]:
    """Get top processes by CPU or memory usage. sort_by can be 'cpu_percent' or 'memory_percent'."""
    return await process_collector.get_top_processes(limit, sort_by)


@mcp.tool()
async def get_network_stats() -> Dict[str, Any]:
    """Get network interface statistics"""
    return await network_collector.get_cached_data()


@mcp.resource("system://live/cpu")
async def live_cpu_resource() -> str:
    """Live CPU usage data updated every 2 seconds.

    Provides a formatted string with current CPU metrics:
    - CPU model and usage percentage
    - Number of CPU cores
    - Current CPU frequency

    This resource is ideal for continuous monitoring scenarios
    where you want to track CPU status without making repeated tool calls."""
    data = await cpu_collector.get_cached_data()
    processor_name = data.get('processor_name', 'Unknown CPU')
    # Shorten processor name if too long
    if len(processor_name) > 30:
        processor_name = processor_name[:27] + "..."
    return f"{processor_name} | Usage: {data.get('usage_percent', 0):.1f}% | Cores: {data.get('core_count', 0)} | Freq: {data.get('frequency_current', 0):.0f} MHz"


@mcp.resource("system://live/memory")
async def live_memory_resource() -> str:
    """Live memory usage data updated every 2 seconds.

    Provides a formatted string showing:
    - Used memory in GB
    - Total memory in GB
    - Memory usage percentage

    This resource is useful for monitoring memory usage trends
    and getting quick memory status updates."""
    data = await memory_collector.get_cached_data()
    return f"Memory: {data.get('used_gb', 0):.1f}GB / {data.get('total_gb', 0):.1f}GB ({data.get('usage_percent', 0):.1f}%)"


@mcp.resource("system://config")
async def system_config_resource() -> str:
    """Static system configuration and hardware information.

    Provides system specifications including:
    - Operating system name and version
    - System architecture (x86_64, ARM64, etc.)
    - Hostname
    - CPU model, vendor, and configuration
    - System uptime since last boot

    This resource is ideal for getting system specifications
    and understanding the hardware/OS environment."""
    system_data = await system_collector.get_cached_data()
    cpu_data = await cpu_collector.get_cached_data()

    cache_info = ""
    if cpu_data.get('cache_sizes'):
        cache_parts = []
        for level, size in cpu_data['cache_sizes'].items():
            if size >= 1024:
                cache_parts.append(f"{level}: {size / 1024:.1f}MB")
            else:
                cache_parts.append(f"{level}: {size}KB")
        cache_info = f"\nCache: {', '.join(cache_parts)}"

    return f"""System Configuration:
OS: {system_data.get('platform', 'N/A')} {system_data.get('architecture', 'N/A')}
Hostname: {system_data.get('hostname', 'N/A')}
CPU: {cpu_data.get('processor_name', 'Unknown')}
Vendor: {cpu_data.get('vendor', 'Unknown')}
Cores: {cpu_data.get('core_count', 0)} physical, {cpu_data.get('thread_count', 0)} logical{cache_info}
Uptime: {system_data.get('uptime_seconds', 0)} seconds
"""


# Phase 1: Performance Enhancement MCP Tools
@mcp.tool()
async def get_io_performance() -> IOPerformanceInfo:
    """Get detailed I/O performance metrics including read/write rates and per-disk statistics.
    
    Returns comprehensive I/O performance data:
    - Read/write bytes and operations per second
    - Average I/O response times
    - Disk busy time percentage
    - Per-disk detailed statistics
    
    This tool is essential for identifying I/O bottlenecks and storage performance issues."""
    data = await io_performance_collector.get_cached_data()
    return IOPerformanceInfo(**data)


@mcp.tool()
async def get_system_load() -> SystemLoadInfo:
    """Get system load averages and process statistics.
    
    Returns system performance metrics:
    - Load averages (1, 5, 15 minutes) on Unix-like systems
    - Context switches and interrupts per second
    - Running and blocked process counts
    - System boot time
    
    This tool helps identify system performance bottlenecks and resource contention."""
    data = await system_load_collector.get_cached_data()
    return SystemLoadInfo(**data)


@mcp.tool()
async def get_enhanced_memory_info() -> EnhancedMemoryInfo:
    """Get detailed memory statistics beyond basic usage.
    
    Returns enhanced memory metrics:
    - Basic memory usage (total, used, available, swap)
    - Buffer and cache memory details (Linux)
    - Active and inactive memory breakdown
    - Page faults and swap activity rates
    
    This tool provides deep insight into memory utilization patterns and performance."""
    data = await enhanced_memory_collector.get_cached_data()
    return EnhancedMemoryInfo(**data)


@mcp.tool()
async def get_enhanced_network_stats() -> List[EnhancedNetworkInfo]:
    """Get detailed network statistics with performance metrics.
    
    Returns comprehensive network data for each interface:
    - Current transfer rates (bytes/packets per second)
    - Total transfer statistics
    - Error and drop counters
    - Interface speed, MTU, and status
    
    This tool is crucial for network performance monitoring and troubleshooting."""
    data = await enhanced_network_collector.get_cached_data()
    enhanced_interfaces = data.get('enhanced_interfaces', [])
    return [EnhancedNetworkInfo(**interface) for interface in enhanced_interfaces]


@mcp.tool()
async def get_performance_snapshot() -> SystemPerformanceSnapshot:
    """Get complete system performance snapshot with all Phase 1 enhancements.
    
    Returns a comprehensive performance overview including:
    - I/O performance metrics
    - System load and process statistics  
    - Enhanced memory analysis
    - Detailed network performance data
    
    This tool provides a complete performance picture for system analysis and optimization."""
    logger.debug("Collecting performance snapshot")
    try:
        # Gather all performance data in parallel
        io_data, load_data, memory_data, network_data = await asyncio.gather(
            io_performance_collector.get_cached_data(),
            system_load_collector.get_cached_data(),
            enhanced_memory_collector.get_cached_data(),
            enhanced_network_collector.get_cached_data()
        )
        
        # Convert network data
        enhanced_interfaces = network_data.get('enhanced_interfaces', [])
        network_list = [EnhancedNetworkInfo(**interface) for interface in enhanced_interfaces]
        
        snapshot = SystemPerformanceSnapshot(
            io_performance=IOPerformanceInfo(**io_data),
            system_load=SystemLoadInfo(**load_data),
            enhanced_memory=EnhancedMemoryInfo(**memory_data),
            enhanced_network=network_list
        )
        
        logger.info("Performance snapshot collected successfully")
        return snapshot
        
    except Exception as e:
        logger.error(f"Failed to collect performance snapshot: {e}")
        raise


@mcp.tool()
async def monitor_io_performance(duration_seconds: int = 5) -> Dict[str, Any]:
    """Monitor I/O performance over a specified duration.
    
    Args:
        duration_seconds: Duration to monitor I/O performance (default: 5 seconds)
    
    Returns monitoring data with:
    - Average, minimum, and maximum I/O rates
    - Sample data for trend analysis
    - Per-disk performance breakdown
    
    This tool helps identify I/O performance patterns and bottlenecks over time."""
    logger.info(f"Starting I/O performance monitoring for {duration_seconds} seconds")
    samples = []
    
    for i in range(duration_seconds):
        data = await io_performance_collector.collect_data()
        sample = {
            "timestamp": datetime.now().isoformat(),
            "read_bytes_per_sec": data['read_bytes_per_sec'],
            "write_bytes_per_sec": data['write_bytes_per_sec'],
            "total_bytes_per_sec": data['read_bytes_per_sec'] + data['write_bytes_per_sec'],
            "busy_time_percent": data['busy_time_percent']
        }
        samples.append(sample)
        logger.debug(f"I/O sample {i + 1}/{duration_seconds}: {sample['total_bytes_per_sec']:.1f} bytes/sec")
        
        if i < duration_seconds - 1:  # Don't sleep after the last sample
            await asyncio.sleep(1)
    
    # Calculate statistics
    total_rates = [sample['total_bytes_per_sec'] for sample in samples]
    busy_times = [sample['busy_time_percent'] for sample in samples]
    
    result = {
        "duration_seconds": duration_seconds,
        "io_rates": {
            "average_bytes_per_sec": sum(total_rates) / len(total_rates) if total_rates else 0,
            "minimum_bytes_per_sec": min(total_rates) if total_rates else 0,
            "maximum_bytes_per_sec": max(total_rates) if total_rates else 0
        },
        "busy_time": {
            "average_percent": sum(busy_times) / len(busy_times) if busy_times else 0,
            "minimum_percent": min(busy_times) if busy_times else 0,
            "maximum_percent": max(busy_times) if busy_times else 0
        },
        "samples": samples
    }
    
    logger.info(f"I/O monitoring complete. Average rate: {result['io_rates']['average_bytes_per_sec']:.1f} bytes/sec")
    return result


# Phase 1: Performance Enhancement Resources
@mcp.resource("system://performance/io")
async def live_io_performance_resource() -> str:
    """Live I/O performance data updated every 2 seconds.
    
    Provides formatted I/O performance metrics:
    - Current read/write rates in MB/s
    - I/O operations per second
    - Disk busy time percentage
    
    This resource is ideal for real-time I/O performance monitoring."""
    data = await io_performance_collector.get_cached_data()
    
    read_mb_s = data.get('read_bytes_per_sec', 0) / (1024 * 1024)
    write_mb_s = data.get('write_bytes_per_sec', 0) / (1024 * 1024)
    read_ops = data.get('read_count_per_sec', 0)
    write_ops = data.get('write_count_per_sec', 0)
    busy_percent = data.get('busy_time_percent', 0)
    
    return f"I/O Performance: Read {read_mb_s:.1f}MB/s ({read_ops:.1f} ops/s) | Write {write_mb_s:.1f}MB/s ({write_ops:.1f} ops/s) | Busy {busy_percent:.1f}%"


@mcp.resource("system://performance/load")
async def live_system_load_resource() -> str:
    """Live system load data updated every 2 seconds.
    
    Provides formatted system load metrics:
    - Load averages (when available)
    - Process counts (running/blocked)
    - System activity indicators
    
    This resource provides quick system performance overview."""
    data = await system_load_collector.get_cached_data()
    
    load_info = ""
    if data.get('load_average_1m') is not None:
        load_info = f"Load: {data['load_average_1m']:.2f}, {data['load_average_5m']:.2f}, {data['load_average_15m']:.2f} | "
    
    processes_info = f"Processes: {data.get('processes_running', 0)} running, {data.get('processes_blocked', 0)} blocked"
    
    return f"System Load: {load_info}{processes_info}"


@mcp.resource("system://performance/network")
async def live_network_performance_resource() -> str:
    """Live network performance data updated every 2 seconds.
    
    Provides formatted network performance summary:
    - Active interface transfer rates
    - Total network activity
    - Interface status overview
    
    This resource gives quick network performance visibility."""
    data = await enhanced_network_collector.get_cached_data()
    interfaces = data.get('enhanced_interfaces', [])
    
    if not interfaces:
        return "Network: No active interfaces detected"
    
    # Find the most active interface
    most_active = max(interfaces, key=lambda x: x['bytes_sent_per_sec'] + x['bytes_recv_per_sec'])
    
    total_sent = sum(iface['bytes_sent_per_sec'] for iface in interfaces if iface['is_up']) / (1024 * 1024)
    total_recv = sum(iface['bytes_recv_per_sec'] for iface in interfaces if iface['is_up']) / (1024 * 1024)
    active_count = sum(1 for iface in interfaces if iface['is_up'])
    
    return f"Network: {active_count} active interfaces | Total: ↑{total_sent:.1f}MB/s ↓{total_recv:.1f}MB/s | Active: {most_active['interface_name']}"


if __name__ == "__main__":
    mcp.run()
