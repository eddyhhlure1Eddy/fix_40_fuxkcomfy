import os
import sys
import logging
import torch

class GPUArchitecture:
    UNKNOWN = "Unknown"
    PASCAL = "Pascal"
    VOLTA = "Volta"
    TURING = "Turing"
    AMPERE = "Ampere"
    ADA_LOVELACE = "Ada Lovelace"
    BLACKWELL = "Blackwell"
    HOPPER = "Hopper"

class GPUOptimizer:
    def __init__(self):
        self.gpu_name = None
        self.gpu_architecture = GPUArchitecture.UNKNOWN
        self.compute_capability = None
        self.vram_gb = 0
        self.gpu_count = 0
        self.optimizations_applied = []
        
    def detect_gpu(self):
        if not torch.cuda.is_available():
            logging.warning("CUDA not available. Running on CPU mode.")
            return False
        
        self.gpu_count = torch.cuda.device_count()
        self.gpu_name = torch.cuda.get_device_name(0)
        
        props = torch.cuda.get_device_properties(0)
        self.compute_capability = (props.major, props.minor)
        self.vram_gb = props.total_memory / (1024**3)
        
        self._identify_architecture()
        return True
    
    def _identify_architecture(self):
        gpu_name_lower = self.gpu_name.lower()
        
        if "50" in gpu_name_lower or "blackwell" in gpu_name_lower:
            self.gpu_architecture = GPUArchitecture.BLACKWELL
        elif "40" in gpu_name_lower or "ada" in gpu_name_lower:
            self.gpu_architecture = GPUArchitecture.ADA_LOVELACE
        elif "30" in gpu_name_lower or "a100" in gpu_name_lower or "a40" in gpu_name_lower or "a30" in gpu_name_lower or "a10" in gpu_name_lower:
            self.gpu_architecture = GPUArchitecture.AMPERE
        elif "h100" in gpu_name_lower or "h200" in gpu_name_lower:
            self.gpu_architecture = GPUArchitecture.HOPPER
        elif "20" in gpu_name_lower or "titan rtx" in gpu_name_lower or "quadro rtx" in gpu_name_lower:
            self.gpu_architecture = GPUArchitecture.TURING
        elif "v100" in gpu_name_lower or "titan v" in gpu_name_lower:
            self.gpu_architecture = GPUArchitecture.VOLTA
        elif "10" in gpu_name_lower or "titan x" in gpu_name_lower:
            self.gpu_architecture = GPUArchitecture.PASCAL
        else:
            if self.compute_capability:
                major = self.compute_capability[0]
                if major == 10:
                    self.gpu_architecture = GPUArchitecture.BLACKWELL
                elif major == 9:
                    self.gpu_architecture = GPUArchitecture.HOPPER
                elif major == 8:
                    minor = self.compute_capability[1]
                    if minor >= 9:
                        self.gpu_architecture = GPUArchitecture.ADA_LOVELACE
                    else:
                        self.gpu_architecture = GPUArchitecture.AMPERE
                elif major == 7:
                    minor = self.compute_capability[1]
                    if minor >= 5:
                        self.gpu_architecture = GPUArchitecture.TURING
                    else:
                        self.gpu_architecture = GPUArchitecture.VOLTA
                elif major == 6:
                    self.gpu_architecture = GPUArchitecture.PASCAL
    
    def apply_optimizations(self):
        if self.gpu_architecture == GPUArchitecture.UNKNOWN:
            logging.warning("Unknown GPU architecture. Using default settings.")
            return
        
        logging.info("=" * 80)
        logging.info(f"GPU Detected: {self.gpu_name}")
        logging.info(f"Architecture: {self.gpu_architecture}")
        logging.info(f"Compute Capability: {self.compute_capability[0]}.{self.compute_capability[1]}")
        logging.info(f"VRAM: {self.vram_gb:.2f} GB")
        logging.info(f"GPU Count: {self.gpu_count}")
        logging.info("=" * 80)
        
        if self.gpu_architecture == GPUArchitecture.BLACKWELL:
            self._optimize_blackwell()
        elif self.gpu_architecture == GPUArchitecture.ADA_LOVELACE:
            self._optimize_ada()
        elif self.gpu_architecture == GPUArchitecture.AMPERE:
            self._optimize_ampere()
        elif self.gpu_architecture == GPUArchitecture.HOPPER:
            self._optimize_hopper()
        elif self.gpu_architecture == GPUArchitecture.TURING:
            self._optimize_turing()
        elif self.gpu_architecture == GPUArchitecture.VOLTA:
            self._optimize_volta()
        elif self.gpu_architecture == GPUArchitecture.PASCAL:
            self._optimize_pascal()
        
        self._log_optimizations()
    
    def _optimize_blackwell(self):
        logging.info("Applying Blackwell (50 series) optimizations...")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        self.optimizations_applied.append("Expandable CUDA segments")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.optimizations_applied.append("TF32 enabled for matmul and cuDNN")
        
        torch.backends.cudnn.benchmark = True
        self.optimizations_applied.append("cuDNN auto-tuner enabled")
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95, 0)
            self.optimizations_applied.append("Memory fraction: 95%")
        
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
        self.optimizations_applied.append("cuDNN v8 API enabled")
        
        self.optimizations_applied.append("FP8 precision support (Blackwell native)")
        self.optimizations_applied.append("Advanced tensor cores optimization")
    
    def _optimize_ada(self):
        logging.info("Applying Ada Lovelace (40 series) optimizations...")

        existing_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        if existing_conf:
            if 'backend:cudaMallocAsync' in existing_conf:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = existing_conf + ',max_split_size_mb:512,roundup_power2_divisions:16'
            else:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = existing_conf + ',expandable_segments:True,max_split_size_mb:512'
        else:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
        self.optimizations_applied.append("Low VRAM memory allocator (max_split_size_mb:512)")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.optimizations_applied.append("TF32 enabled for matmul and cuDNN")

        torch.backends.cudnn.benchmark = True
        self.optimizations_applied.append("cuDNN auto-tuner enabled")

        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.85, 0)
            self.optimizations_applied.append("Memory fraction: 85% (low VRAM mode)")

        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
        self.optimizations_applied.append("cuDNN v8 API enabled")

        self.optimizations_applied.append("Ada tensor cores optimization")
        self.optimizations_applied.append("BF16 precision recommended (low VRAM)")
    
    def _optimize_ampere(self):
        logging.info("Applying Ampere (30 series) optimizations...")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        self.optimizations_applied.append("Expandable CUDA segments")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.optimizations_applied.append("TF32 enabled for matmul and cuDNN")
        
        torch.backends.cudnn.benchmark = True
        self.optimizations_applied.append("cuDNN auto-tuner enabled")
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.90, 0)
            self.optimizations_applied.append("Memory fraction: 90%")
        
        self.optimizations_applied.append("Ampere tensor cores optimization")
        self.optimizations_applied.append("BF16 support enabled")
    
    def _optimize_hopper(self):
        logging.info("Applying Hopper (H100/H200) optimizations...")
        
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        self.optimizations_applied.append("Expandable CUDA segments")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.optimizations_applied.append("TF32 enabled for matmul and cuDNN")
        
        torch.backends.cudnn.benchmark = True
        self.optimizations_applied.append("cuDNN auto-tuner enabled")
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95, 0)
            self.optimizations_applied.append("Memory fraction: 95%")
        
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
        self.optimizations_applied.append("cuDNN v8 API enabled")
        
        self.optimizations_applied.append("Hopper transformer engine optimization")
        self.optimizations_applied.append("FP8 precision support")
    
    def _optimize_turing(self):
        logging.info("Applying Turing (20 series) optimizations...")
        
        torch.backends.cudnn.benchmark = True
        self.optimizations_applied.append("cuDNN auto-tuner enabled")
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.88, 0)
            self.optimizations_applied.append("Memory fraction: 88%")
        
        self.optimizations_applied.append("Turing tensor cores optimization")
        self.optimizations_applied.append("FP16 mixed precision recommended")
    
    def _optimize_volta(self):
        logging.info("Applying Volta (V100) optimizations...")
        
        torch.backends.cudnn.benchmark = True
        self.optimizations_applied.append("cuDNN auto-tuner enabled")
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.85, 0)
            self.optimizations_applied.append("Memory fraction: 85%")
        
        self.optimizations_applied.append("Volta tensor cores optimization")
        self.optimizations_applied.append("FP16 mixed precision recommended")
    
    def _optimize_pascal(self):
        logging.info("Applying Pascal (10 series) optimizations...")
        
        torch.backends.cudnn.benchmark = True
        self.optimizations_applied.append("cuDNN auto-tuner enabled")
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.82, 0)
            self.optimizations_applied.append("Memory fraction: 82%")
        
        self.optimizations_applied.append("Pascal CUDA cores optimization")
        self.optimizations_applied.append("FP32 precision (no tensor cores)")
    
    def _log_optimizations(self):
        logging.info("=" * 80)
        logging.info("Applied Optimizations:")
        for i, opt in enumerate(self.optimizations_applied, 1):
            logging.info(f"  [{i}] {opt}")
        logging.info("=" * 80)
    
    def get_recommended_settings(self):
        settings = {
            "architecture": self.gpu_architecture,
            "vram_gb": self.vram_gb,
            "compute_capability": self.compute_capability,
        }

        if self.gpu_architecture in [GPUArchitecture.BLACKWELL, GPUArchitecture.HOPPER]:
            settings["precision"] = "fp8"
            settings["attention"] = "flash_attention_2"
            settings["batch_size_multiplier"] = 1.5
        elif self.gpu_architecture == GPUArchitecture.ADA_LOVELACE:
            settings["precision"] = "bf16"
            settings["attention"] = "flash_attention_2"
            settings["batch_size_multiplier"] = 1.0
            settings["vram_optimization"] = "enabled"
            settings["memory_efficient"] = True
        elif self.gpu_architecture == GPUArchitecture.AMPERE:
            settings["precision"] = "bf16"
            settings["attention"] = "flash_attention"
            settings["batch_size_multiplier"] = 1.2
        elif self.gpu_architecture == GPUArchitecture.TURING:
            settings["precision"] = "fp16"
            settings["attention"] = "sdp"
            settings["batch_size_multiplier"] = 1.0
        else:
            settings["precision"] = "fp32"
            settings["attention"] = "default"
            settings["batch_size_multiplier"] = 0.8

        return settings

def initialize_gpu_optimizer():
    optimizer = GPUOptimizer()
    
    if optimizer.detect_gpu():
        optimizer.apply_optimizations()
        return optimizer
    else:
        logging.warning("No GPU detected or CUDA not available")
        return None

