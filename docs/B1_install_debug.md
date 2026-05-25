# MuJoCo Leap Hand Project: Installation & Debug Log

## Environment Setup

**Instance**: AutoDL RTX 5090
- CUDA: 13.0 (driver: 580.76)
- OS: Ubuntu 22.04
- Python: 3.12 via miniconda

## Installation Challenges & Solutions

### Issue 1: Conda Environment Creation Failed

**Error**: Conda mirror unavailable during environment creation
```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url ...
```

**Solution**: Switched from conda env to Python venv
```bash
python -m venv /root/autodl-tmp/mujoco_playground/.venv
source /root/autodl-tmp/mujoco_playground/.venv/bin/activate
```

### Issue 2: Pip Download Throttling

**Error**: Slow or failed pip package downloads

**Solution**: Enable AutoDL network turbo before installing packages
```bash
source /etc/network_turbo
pip install jax[cuda12]
```

### Issue 3: JAX CUDA Library Conflicts (Critical)

**Error**: Runtime library loading failure when importing JAX
```
ImportError: libnvrtc.so.12: cannot open shared object file
```

**Root Cause**: AutoDL sets `LD_LIBRARY_PATH` globally, which conflicts with JAX's bundled CUDA libraries

**Solution**: Unset LD_LIBRARY_PATH before any JAX operations
```bash
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.device_count())"
```

### Issue 4: Blackwell GPU Matmul Precision

**Recommendation**: Set JAX matmul precision for Blackwell GPUs
```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

### Issue 5: brax 0.14.2 Incompatible with JAX 0.10+

**Error**: `jax.device_put_replicated` removed in JAX 0.10
```
AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead.
```

**Root Cause**: brax 0.14.2 (latest PyPI) still uses `jax.device_put_replicated` in `brax/training/agents/ppo/train.py:756` and 3 other files. No fixed version available on PyPI as of 2026-05-25.

**Solution**: Monkey-patch via site-packages `.pth` file:
- Created `jax_compat_patch.py` in site-packages that restores `jax.device_put_replicated`
- Created `jax_compat_patch.pth` to auto-import on Python startup
- The shim adds a leading batch dim (size=1) to match brax's `_unpmap` which does `.squeeze(0)`

### Issue 6: Aliyun pip mirror missing jax-cuda12-plugin

**Error**: `pip install jax[cuda12]` fails when using default Aliyun mirror
```
ERROR: No matching distribution found for jax-cuda12-plugin
```

**Solution**: Always specify PyPI directly for JAX CUDA packages
```bash
pip install "jax[cuda12]" --index-url https://pypi.org/simple
```

## Package Installation

Installed via venv + pip (all from PyPI, not Aliyun mirror):
- JAX 0.10.1 with cuda12 plugin + pjrt
- jaxlib 0.10.1
- brax 0.14.2 (with monkey-patch for JAX 0.10 compat)
- MuJoCo 3.8.1 + MuJoCo MJX 3.8.1
- mujoco_playground 0.2.0 from source: `pip install -e .`
- wandb, tensorboardX, mediapy

## Storage Configuration

- Data disk: `/root/autodl-tmp/` (large capacity)
- System disk: 30GB overlay filesystem
- All project/environment files on data disk to avoid space exhaustion

## Activation Checklist

Before running any MuJoCo/JAX code:
```bash
source /root/autodl-tmp/mujoco_playground/.venv/bin/activate
unset LD_LIBRARY_PATH
export JAX_DEFAULT_MATMUL_PRECISION=highest
source /etc/network_turbo  # if installing new packages
```
