#!/usr/bin/env bash
# 项目 B - 步骤 2：安装 MuJoCo Playground
# 原则：大文件/cache/venv 全部在 /root/autodl-tmp，不撑系统盘 (overlay 30G)
set -euo pipefail

export JAX_DEFAULT_MATMUL_PRECISION=highest
DATA=/root/autodl-tmp
LOG=$DATA/mjx_setup.log
exec > >(tee -a "$LOG") 2>&1

check_disk() {
  local avail_sys avail_data
  avail_sys=$(df -BG / | awk 'NR==2 {print $4}' | tr -d G)
  avail_data=$(df -BG "$DATA" | awk 'NR==2 {print $4}' | tr -d G)
  echo "[disk] / free=${avail_sys}G  ${DATA} free=${avail_data}G"
  if [[ "$avail_sys" -lt 3 ]]; then
    echo "ERROR: system disk / free < 3G — abort"
    exit 1
  fi
  if [[ "$avail_data" -lt 8 ]]; then
    echo "ERROR: data disk free < 8G — abort"
    exit 1
  fi
}

echo "=== B0 setup start: $(date) ==="
check_disk
source /etc/network_turbo 2>/dev/null || true

# 临时文件 / 缓存 / 模型下载 全部放数据盘
export TMPDIR=$DATA/tmp_mjx
export PIP_CACHE_DIR=$DATA/pip_cache_mjx
export HF_HOME=$DATA/hf_cache
export XDG_CACHE_HOME=$DATA/xdg_cache
export MUJOCO_PY_FORCE_CPU=0
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

cd "$DATA"
if [[ ! -d mujoco_playground ]]; then
  git clone --depth 1 https://github.com/google-deepmind/mujoco_playground.git
fi
cd mujoco_playground

export PATH="/root/miniconda3/bin:$PATH"
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_TRUSTED_HOST=pypi.org

# venv 在数据盘项目目录内（已在 autodl-tmp）
if [[ ! -d .venv ]]; then
  /root/miniconda3/bin/python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
check_disk

pip install -U pip wheel -q

# 若 jax 已装好则跳过（支持断点续装）
if ! python -c "import jax; assert jax.default_backend()=='gpu'" 2>/dev/null; then
  echo "Installing jax[cuda12] ..."
  pip install -U "jax[cuda12]"
fi
check_disk

unset LD_LIBRARY_PATH || true
python -c "import jax; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', jax.devices())"

if ! python -c "import mujoco_playground" 2>/dev/null; then
  echo "Installing mujoco_playground ..."
  pip install -e ".[all]" 2>/dev/null || pip install -e .
fi
python -c "import mujoco_playground; print('mujoco_playground OK')"

cat > "$DATA/mjx_activate.sh" << 'ACTIVATE'
#!/usr/bin/env bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
export TMPDIR=/root/autodl-tmp/tmp_mjx
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache_mjx
export HF_HOME=/root/autodl-tmp/hf_cache
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache
unset LD_LIBRARY_PATH 2>/dev/null || true
source /root/autodl-tmp/mujoco_playground/.venv/bin/activate
cd /root/autodl-tmp/mujoco_playground
ACTIVATE
chmod +x "$DATA/mjx_activate.sh"

python - << 'PY'
from mujoco_playground import registry
try:
    names = sorted(registry.ALL_ENVS)
except AttributeError:
    names = sorted(registry._envs.keys())
for n in names:
    low = n.lower()
    if any(k in low for k in ("leap", "cube", "hand", "reorient", "panda")):
        print(n)
PY

check_disk
du -sh .venv "$PIP_CACHE_DIR" "$HF_HOME" 2>/dev/null || true
echo "=== B0 setup done: $(date) ==="
