import torch
import logging
import os
from enum import Enum
from typing import Optional, Dict, Any, Tuple

class MatrixPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    FP16_FAST = "fp16_fast"
    BF16 = "bf16"
    TF32 = "tf32"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"

class MatrixOptimizer:
    def __init__(self, gpu_optimizer=None):
        self.gpu_optimizer = gpu_optimizer
        self.supported_precisions = []
        self.optimal_precision = None
        self.matmul_precision = None
        self.capabilities = {}
        
        self._detect_capabilities()
        self._select_optimal_precision()
    
    def _detect_capabilities(self):
        logging.info("Detecting matrix multiplication capabilities...")
        
        self.capabilities['cuda_available'] = torch.cuda.is_available()
        
        if not self.capabilities['cuda_available']:
            logging.warning("CUDA not available - using CPU mode")
            self.supported_precisions = [MatrixPrecision.FP32]
            return
        
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = (device_props.major, device_props.minor)
        
        self.capabilities['compute_capability'] = compute_capability
        self.capabilities['tensor_cores'] = compute_capability[0] >= 7
        
        self.capabilities['fp32'] = True
        self.supported_precisions.append(MatrixPrecision.FP32)
        
        self.capabilities['fp16'] = True
        self.supported_precisions.append(MatrixPrecision.FP16)
        self.supported_precisions.append(MatrixPrecision.FP16_FAST)
        
        if compute_capability[0] >= 8:
            self.capabilities['bf16'] = True
            self.supported_precisions.append(MatrixPrecision.BF16)
            
            self.capabilities['tf32'] = True
            self.supported_precisions.append(MatrixPrecision.TF32)
        
        if compute_capability[0] >= 9 or (compute_capability[0] == 8 and compute_capability[1] >= 9):
            self.capabilities['fp8'] = True
            self.supported_precisions.append(MatrixPrecision.FP8_E4M3)
            self.supported_precisions.append(MatrixPrecision.FP8_E5M2)
        
        self.capabilities['cudnn_enabled'] = torch.backends.cudnn.enabled
        self.capabilities['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.enabled else None
        
        try:
            self.capabilities['flash_attention'] = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        except:
            self.capabilities['flash_attention'] = False
        
        logging.info(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        logging.info(f"Tensor Cores: {self.capabilities['tensor_cores']}")
        logging.info(f"Supported Precisions: {[p.value for p in self.supported_precisions]}")
    
    def _select_optimal_precision(self):
        if not self.capabilities['cuda_available']:
            self.optimal_precision = MatrixPrecision.FP32
            return

        compute_capability = self.capabilities['compute_capability']

        if compute_capability[0] >= 10:
            self.optimal_precision = MatrixPrecision.FP16_FAST
        elif compute_capability[0] == 9:
            self.optimal_precision = MatrixPrecision.FP16_FAST
        elif compute_capability[0] == 8 and compute_capability[1] >= 9:
            self.optimal_precision = MatrixPrecision.BF16
        elif compute_capability[0] == 8:
            self.optimal_precision = MatrixPrecision.BF16
        elif compute_capability[0] == 7:
            self.optimal_precision = MatrixPrecision.FP16_FAST
        else:
            self.optimal_precision = MatrixPrecision.FP32

        logging.info(f"Optimal Precision Selected: {self.optimal_precision.value}")
    
    def configure_matmul_precision(self, precision: Optional[MatrixPrecision] = None):
        if precision is None:
            precision = self.optimal_precision
        
        if precision not in self.supported_precisions:
            logging.warning(f"{precision.value} not supported, falling back to {self.optimal_precision.value}")
            precision = self.optimal_precision
        
        self.matmul_precision = precision
        
        if precision == MatrixPrecision.TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            logging.info("Configured: TF32 precision for matmul")
        
        elif precision in [MatrixPrecision.FP16, MatrixPrecision.FP16_FAST]:
            torch.backends.cuda.matmul.allow_tf32 = False
            if precision == MatrixPrecision.FP16_FAST:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = False
                logging.info("Configured: FP16 Fast precision (cuDNN benchmark enabled)")
            else:
                logging.info("Configured: FP16 precision")
        
        elif precision == MatrixPrecision.BF16:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("Configured: BF16 precision")
        
        elif precision in [MatrixPrecision.FP8_E4M3, MatrixPrecision.FP8_E5M2]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info(f"Configured: {precision.value} precision")
        
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logging.info("Configured: FP32 precision")
        
        return precision
    
    def get_dtype_for_precision(self, precision: MatrixPrecision) -> torch.dtype:
        dtype_map = {
            MatrixPrecision.FP32: torch.float32,
            MatrixPrecision.FP16: torch.float16,
            MatrixPrecision.FP16_FAST: torch.float16,
            MatrixPrecision.BF16: torch.bfloat16,
            MatrixPrecision.TF32: torch.float32,
            MatrixPrecision.FP8_E4M3: torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16,
            MatrixPrecision.FP8_E5M2: torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else torch.float16,
        }
        return dtype_map.get(precision, torch.float32)
    
    def optimize_matmul(self, a: torch.Tensor, b: torch.Tensor,
                       precision: Optional[MatrixPrecision] = None) -> torch.Tensor:
        if precision is None:
            precision = self.matmul_precision or self.optimal_precision

        dtype = self.get_dtype_for_precision(precision)

        original_dtype = a.dtype

        try:
            a_converted = a.to(dtype)
            b_converted = b.to(dtype)

            result = torch.matmul(a_converted, b_converted)

            if precision in [MatrixPrecision.FP8_E4M3, MatrixPrecision.FP8_E5M2]:
                result = result.to(torch.float16)

            return result.to(original_dtype)
        except Exception as e:
            logging.warning(f"Failed to use {precision.value}, falling back to FP16: {e}")
            a_fp16 = a.to(torch.float16)
            b_fp16 = b.to(torch.float16)
            result = torch.matmul(a_fp16, b_fp16)
            return result.to(original_dtype)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            'optimal_precision': self.optimal_precision.value if self.optimal_precision else None,
            'current_precision': self.matmul_precision.value if self.matmul_precision else None,
            'supported_precisions': [p.value for p in self.supported_precisions],
            'capabilities': self.capabilities,
            'tf32_enabled': torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            'cudnn_benchmark': torch.backends.cudnn.benchmark if torch.cuda.is_available() else False,
        }
    
    def benchmark_precision(self, size: int = 4096, iterations: int = 100) -> Dict[str, float]:
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, skipping benchmark")
            return {}
        
        results = {}
        device = torch.device('cuda')
        
        logging.info(f"Benchmarking matrix multiplication ({size}x{size})...")
        
        for precision in self.supported_precisions:
            dtype = self.get_dtype_for_precision(precision)
            
            try:
                a = torch.randn(size, size, dtype=dtype, device=device)
                b = torch.randn(size, size, dtype=dtype, device=device)
                
                torch.cuda.synchronize()
                
                import time
                start = time.perf_counter()
                
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                
                elapsed = (end - start) / iterations * 1000
                results[precision.value] = elapsed
                
                logging.info(f"  {precision.value}: {elapsed:.3f} ms/iteration")
                
            except Exception as e:
                logging.warning(f"  {precision.value}: Failed - {e}")
                results[precision.value] = float('inf')
        
        return results

class MatrixRouter:
    def __init__(self, matrix_optimizer: MatrixOptimizer):
        self.optimizer = matrix_optimizer
        self.routes = {}
        self._setup_routes()
    
    def _setup_routes(self):
        self.routes = {
            'fp32': lambda a, b: self.optimizer.optimize_matmul(a, b, MatrixPrecision.FP32),
            'fp16': lambda a, b: self.optimizer.optimize_matmul(a, b, MatrixPrecision.FP16),
            'fp16_fast': lambda a, b: self.optimizer.optimize_matmul(a, b, MatrixPrecision.FP16_FAST),
            'bf16': lambda a, b: self.optimizer.optimize_matmul(a, b, MatrixPrecision.BF16),
            'tf32': lambda a, b: self.optimizer.optimize_matmul(a, b, MatrixPrecision.TF32),
            'fp8_e4m3': lambda a, b: self.optimizer.optimize_matmul(a, b, MatrixPrecision.FP8_E4M3),
            'fp8_e5m2': lambda a, b: self.optimizer.optimize_matmul(a, b, MatrixPrecision.FP8_E5M2),
            'auto': lambda a, b: self.optimizer.optimize_matmul(a, b, None),
        }
    
    def route(self, precision: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if precision not in self.routes:
            logging.warning(f"Unknown precision '{precision}', using auto")
            precision = 'auto'
        
        return self.routes[precision](a, b)
    
    def get_available_routes(self) -> list:
        return list(self.routes.keys())

def initialize_matrix_optimizer(gpu_optimizer=None):
    optimizer = MatrixOptimizer(gpu_optimizer)
    optimizer.configure_matmul_precision()
    
    router = MatrixRouter(optimizer)
    
    logging.info("=" * 80)
    logging.info("Matrix Optimization System Initialized")
    logging.info(f"Optimal Precision: {optimizer.optimal_precision.value}")
    logging.info(f"Available Routes: {router.get_available_routes()}")
    logging.info("=" * 80)
    
    return optimizer, router

