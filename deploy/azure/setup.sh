#!/usr/bin/env bash
# Prepara una VM GPU de Azure (Ubuntu 22.04/24.04 con drivers NVIDIA) para NAVROS.
#
#   ./deploy/azure/setup.sh             # entorno + verificación de CUDA
#   ./deploy/azure/setup.sh --service   # además, automejora continua como servicio systemd
#
# Variables opcionales: NAVROS_PRESET (base), NAVROS_RUN (runs/navros), NAVROS_MODE (verifier)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PRESET="${NAVROS_PRESET:-base}"
RUN="${NAVROS_RUN:-runs/navros}"
MODE="${NAVROS_MODE:-verifier}"
cd "$REPO_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No se encontró nvidia-smi. Usa la imagen Ubuntu-HPC o instala la extensión de drivers:"
  echo "  az vm extension set -g <grupo> --vm-name <vm> --name NvidiaGpuDriverLinux --publisher Microsoft.HpcCompute"
  exit 1
fi
nvidia-smi

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git tmux
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # el torch de PyPI para Linux ya incluye CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible'; print('GPU:', torch.cuda.get_device_name(0))"

if [ ! -f "$RUN/model.pt" ]; then
  python -m navros init --preset "$PRESET" --mode "$MODE" --run "$RUN"
fi

if [ "${1:-}" = "--service" ]; then
  sed -e "s|__REPO__|$REPO_DIR|g" -e "s|__USER__|$USER|g" -e "s|__RUN__|$RUN|g" \
    deploy/azure/navros.service | sudo tee /etc/systemd/system/navros.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now navros
  echo "Servicio activo. Registro en vivo:  journalctl -u navros -f"
else
  echo "Listo. Ejecuta, por ejemplo:"
  echo "  . .venv/bin/activate && tmux new -s navros 'python -m navros improve --run $RUN --rounds 0'"
fi
