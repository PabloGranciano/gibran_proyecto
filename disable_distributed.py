# disable_distributed.py
import os
import torch
import torch.distributed as dist

# 1) Quitar variables de entorno que activan distributed
for v in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
          "LOCAL_WORLD_SIZE", "SLURM_JOB_ID", "SLURM_PROCID", "SLURM_NTASKS"]:
    os.environ.pop(v, None)

# 2) Evitar que torch intente usar NCCL o GPUs
# (opcional si quieres forzar CPU)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# 3) Forzar que PyTorch crea que distributed no está disponible/ini.
dist.is_available = lambda: False
dist.is_initialized = lambda: False

# 4) Sobrescribir funciones que podrían inicializar/distribuir
def _fake_init_process_group(*args, **kwargs):
    print("⚠️ distributed.init_process_group() bloqueado por disable_distributed.")
    return None

dist.init_process_group = _fake_init_process_group
dist.get_rank = lambda: 0
dist.get_world_size = lambda: 1
dist.barrier = lambda *a, **k: None

# 5) También interceptar la función enable del paquete dinov2
# (si dinov2 ya la importó, la reemplazamos en runtime)
try:
    import astroclip.astrodino.distributed as d2dist  # si ya está instalado
    d2dist.enable = lambda *args, **kwargs: None
except Exception:
    pass
