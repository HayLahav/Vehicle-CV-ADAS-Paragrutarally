#!/usr/bin/env python3
"""
Jetson Xavier Utilities for Vehicle-CV-ADAS
Provides Jetson-specific optimizations and monitoring functions
"""

import os
import subprocess
import time
import threading
from typing import Dict, Optional
import logging

class JetsonOptimizer:
    """Jetson Xavier performance optimization and monitoring"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.is_jetson = self._detect_jetson()
        self.performance_mode_set = False
        self.monitoring_active = False
        self.stats = {}
        
    def _detect_jetson(self) -> bool:
        """Detect if running on Jetson platform"""
        try:
            with open('/etc/nv_tegra_release') as f:
                return True
        except FileNotFoundError:
            return False
    
    def get_jetson_info(self) -> Dict[str, str]:
        """Get detailed Jetson platform information"""
        info = {}
        
        if not self.is_jetson:
            return {"platform": "Not Jetson"}
        
        try:
            # Get Tegra release info
            with open('/etc/nv_tegra_release') as f:
                tegra_info = f.read().strip()
                info['tegra_release'] = tegra_info
            
            # Try to get more detailed info via jetson_release if available
            try:
                result = subprocess.run(['jetson_release', '--show-all'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['jetson_release'] = result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Get model information
            try:
                with open('/proc/device-tree/model') as f:
                    model = f.read().strip('\x00')
                    info['model'] = model
            except FileNotFoundError:
                pass
            
            # Get CUDA version
            try:
                result = subprocess.run(['nvcc', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'release' in line.lower():
                            info['cuda_version'] = line.strip()
                            break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        except Exception as e:
            self.logger.warning(f"Could not get complete Jetson info: {e}")
        
        return info
    
    def set_performance_mode(self, mode: int = 0) -> bool:
        """
        Set Jetson performance mode
        Args:
            mode: Performance mode (0 = max performance, higher = lower performance)
        Returns:
            bool: Success status
        """
        if not self.is_jetson:
            self.logger.warning("Not running on Jetson platform")
            return False
        
        try:
            # Set nvpmodel
            result = subprocess.run(['sudo', 'nvpmodel', '-m', str(mode)], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.logger.warning(f"nvpmodel failed: {result.stderr}")
                return False
            
            # Enable jetson_clocks
            result = subprocess.run(['sudo', 'jetson_clocks'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.logger.warning(f"jetson_clocks failed: {result.stderr}")
                return False
            
            self.performance_mode_set = True
            self.logger.info(f"Jetson performance mode set to {mode}")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Performance mode setting timed out")
            return False
        except Exception as e:
            self.logger.error(f"Failed to set performance mode: {e}")
            return False
    
    def set_gpu_frequency(self, freq_hz: Optional[int] = None) -> bool:
        """
        Set GPU maximum frequency
        Args:
            freq_hz: Frequency in Hz (None for maximum)
        Returns:
            bool: Success status
        """
        if not self.is_jetson:
            return False
        
        try:
            if freq_hz is None:
                # Set to maximum frequency for Xavier AGX
                freq_hz = 1377000000  # 1377 MHz
            
            cmd = f'echo {freq_hz} > /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq'
            result = subprocess.run(['sudo', 'bash', '-c', cmd], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.logger.info(f"GPU frequency set to {freq_hz/1e6:.0f} MHz")
                return True
            else:
                self.logger.warning(f"Could not set GPU frequency: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.warning(f"GPU frequency setting failed: {e}")
            return False
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        stats = {}
        
        if not self.is_jetson:
            return stats
        
        try:
            # Temperature monitoring
            temp_zones = [
                '/sys/devices/virtual/thermal/thermal_zone0/temp',
                '/sys/devices/virtual/thermal/thermal_zone1/temp',
                '/sys/devices/virtual/thermal/thermal_zone2/temp'
            ]
            
            temperatures = []
            for i, zone in enumerate(temp_zones):
                try:
                    with open(zone, 'r') as f:
                        temp = int(f.read().strip()) / 1000.0
                        temperatures.append(temp)
                        stats[f'temp_zone_{i}'] = temp
                except FileNotFoundError:
                    continue
            
            if temperatures:
                stats['temperature_max'] = max(temperatures)
                stats['temperature_avg'] = sum(temperatures) / len(temperatures)
            
            # Memory usage
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        total_kb = int(line.split()[1])
                        stats['memory_total_mb'] = total_kb // 1024
                    elif 'MemAvailable:' in line:
                        available_kb = int(line.split()[1])
                        stats['memory_available_mb'] = available_kb // 1024
                
                if 'memory_total_mb' in stats and 'memory_available_mb' in stats:
                    stats['memory_used_mb'] = stats['memory_total_mb'] - stats['memory_available_mb']
                    stats['memory_usage_percent'] = (stats['memory_used_mb'] / stats['memory_total_mb']) * 100
            except Exception as e:
                self.logger.warning(f"Memory stats failed: {e}")
            
            # GPU usage (if available)
            try:
                gpu_load_file = '/sys/devices/gpu.0/load'
                if os.path.exists(gpu_load_file):
                    with open(gpu_load_file, 'r') as f:
                        gpu_load = int(f.read().strip())
                        stats['gpu_usage_percent'] = gpu_load / 10  # Convert to percentage
            except Exception:
                pass
            
            # CPU frequencies
            try:
                cpu_freqs = []
                for i in range(8):  # Xavier has up to 8 cores
                    freq_file = f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq'
                    if os.path.exists(freq_file):
                        with open(freq_file, 'r') as f:
                            freq_khz = int(f.read().strip())
                            cpu_freqs.append(freq_khz / 1000)  # Convert to MHz
                
                if cpu_freqs:
                    stats['cpu_freq_avg_mhz'] = sum(cpu_freqs) / len(cpu_freqs)
                    stats['cpu_freq_max_mhz'] = max(cpu_freqs)
            except Exception:
                pass
            
            # Power consumption (if available)
            try:
                power_files = [
                    '/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input',
                    '/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input'
                ]
                
                total_power = 0
                power_readings = 0
                
                for power_file in power_files:
                    if os.path.exists(power_file):
                        try:
                            with open(power_file, 'r') as f:
                                power_mw = int(f.read().strip())
                                total_power += power_mw
                                power_readings += 1
                        except:
                            continue
                
                if power_readings > 0:
                    stats['power_consumption_w'] = total_power / 1000.0
            except Exception:
                pass
                
        except Exception as e:
            self.logger.warning(f"System stats collection failed: {e}")
        
        return stats
    
    def start_monitoring(self, interval: float = 2.0):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                self.stats = self.get_system_stats()
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("System monitoring stopped")
    
    def get_latest_stats(self) -> Dict:
        """Get latest monitoring statistics"""
        return self.stats.copy()
    
    def check_thermal_throttling(self) -> bool:
        """Check if system is thermal throttling"""
        stats = self.get_system_stats()
        
        # Xavier typically throttles around 87°C
        temp_threshold = 85.0
        
        if 'temperature_max' in stats:
            return stats['temperature_max'] > temp_threshold
        
        return False
    
    def optimize_for_inference(self):
        """Apply optimizations specifically for AI inference"""
        if not self.is_jetson:
            self.logger.warning("Not running on Jetson platform")
            return False
        
        success = True
        
        # Set maximum performance mode
        if not self.set_performance_mode(0):
            success = False
        
        # Set GPU to maximum frequency
        if not self.set_gpu_frequency():
            success = False
        
        # Set environment variables for optimization
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        os.environ['CUDA_CACHE_PATH'] = '/tmp/.nv/ComputeCache'
        
        # Ensure GPU memory growth (for TensorFlow if used)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # CUDNN optimizations
        os.environ['CUDNN_BENCHMARK'] = '1'
        
        self.logger.info("Applied Jetson inference optimizations")
        return success
    
    def get_recommended_settings(self) -> Dict[str, str]:
        """Get recommended settings for ADAS application"""
        recommendations = {
            "model_precision": "FP16 for best performance",
            "batch_size": "1 for real-time inference",
            "input_resolution": "640x640 or smaller for YOLO models",
            "memory_optimization": "Enable CUDA memory pooling",
            "thermal_management": "Monitor temperature < 85°C"
        }
        
        if self.is_jetson:
            stats = self.get_system_stats()
            
            if 'memory_total_mb' in stats:
                if stats['memory_total_mb'] < 8000:  # Less than 8GB
                    recommendations["model_size"] = "Use nano or small models (YOLOv5n/s)"
                else:
                    recommendations["model_size"] = "Can use medium models (YOLOv5m)"
            
            if 'temperature_max' in stats and stats['temperature_max'] > 80:
                recommendations["cooling"] = "Consider additional cooling"
        
        return recommendations


class JetsonModelOptimizer:
    """Utilities for optimizing models for Jetson deployment"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def convert_to_tensorrt(self, onnx_path: str, output_path: str, 
                          fp16: bool = True, max_workspace_size: int = 1 << 30) -> bool:
        """
        Convert ONNX model to TensorRT for Jetson
        Args:
            onnx_path: Path to ONNX model
            output_path: Path for TensorRT engine
            fp16: Enable FP16 precision
            max_workspace_size: Maximum workspace size in bytes
        Returns:
            bool: Success status
        """
        try:
            import tensorrt as trt
            
            # Create builder and config
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # Parse ONNX model
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    self.logger.error("Failed to parse ONNX model")
                    return False
            
            # Build engine
            engine = builder.build_engine(network, config)
            if not engine:
                self.logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"TensorRT engine saved to {output_path}")
            return True
            
        except ImportError:
            self.logger.error("TensorRT not available")
            return False
        except Exception as e:
            self.logger.error(f"TensorRT conversion failed: {e}")
            return False
    
    def get_optimal_batch_size(self, model_path: str, input_shape: tuple) -> int:
        """Determine optimal batch size for given model and input shape"""
        # For real-time inference on Jetson, batch size 1 is typically optimal
        return 1
    
    def benchmark_model(self, model_path: str, input_shape: tuple, 
                       iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance on Jetson
        Args:
            model_path: Path to model file
            input_shape: Input tensor shape
            iterations: Number of iterations for benchmarking
        Returns:
            Dict with timing statistics
        """
        try:
            import numpy as np
            from coreEngine import create_engine
            
            # Create engine
            engine = create_engine(model_path)
            
            # Prepare dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                engine.engine_inference(dummy_input)
            
            # Benchmark
            times = []
            for _ in range(iterations):
                start_time = time.time()
                engine.engine_inference(dummy_input)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            stats = {
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'fps': float(1.0 / np.mean(times))
            }
            
            self.logger.info(f"Benchmark results: {stats['fps']:.1f} FPS")
            return stats
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            return {}


# Convenience functions
def setup_jetson_environment(logger=None):
    """Setup optimal Jetson environment for ADAS application"""
    optimizer = JetsonOptimizer(logger)
    
    if not optimizer.is_jetson:
        if logger:
            logger.info("Not running on Jetson, skipping optimization")
        return optimizer
    
    # Get platform info
    info = optimizer.get_jetson_info()
    if logger:
        logger.info(f"Jetson Platform: {info.get('model', 'Unknown')}")
    
    # Apply optimizations
    optimizer.optimize_for_inference()
    
    # Start monitoring
    optimizer.start_monitoring()
    
    return optimizer

def get_jetson_status():
    """Get quick Jetson status summary"""
    optimizer = JetsonOptimizer()
    
    if not optimizer.is_jetson:
        return {"status": "Not Jetson"}
    
    stats = optimizer.get_system_stats()
    
    status = {
        "platform": "Jetson",
        "temperature": stats.get('temperature_max', 'Unknown'),
        "memory_usage": stats.get('memory_usage_percent', 'Unknown'),
        "thermal_throttling": optimizer.check_thermal_throttling()
    }
    
    return status

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    optimizer = setup_jetson_environment()
    
    print("Jetson Status:")
    status = get_jetson_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nRecommended Settings:")
    recommendations = optimizer.get_recommended_settings()
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    # Monitor for 10 seconds
    print("\nMonitoring for 10 seconds...")
    time.sleep(10)
    
    final_stats = optimizer.get_latest_stats()
    print(f"Final temperature: {final_stats.get('temperature_max', 'Unknown')}°C")
    print(f"Memory usage: {final_stats.get('memory_usage_percent', 'Unknown')}%")
    
    optimizer.stop_monitoring()