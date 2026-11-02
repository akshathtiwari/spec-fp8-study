#!/usr/bin/env bash
# Host-side steps for F042's driver-swap intervention, run ON the GCP VM.
# Each subcommand is one step, run separately so each can be verified before
# the next spends money.
#
#   host.sh driver 580.95.05   install (or replace) the NVIDIA driver
#   host.sh docker             Docker + NVIDIA Container Toolkit
#   host.sh build              the Modal-equivalent image (cloud/gcp/Dockerfile)
#   host.sh verify             driver, card, torch, vllm, async flag -- no GPU load
#   host.sh run ARGS...        cloud/gcp/run_in_container.py ARGS
#                              (CPUSET=0 host.sh run ... pins to one vCPU)
#
# The repo is mounted from the VM user's disk into a container that runs as
# root, so git refuses it ("dubious ownership", exit 128) and boot_variance's
# own `git pull` fails before measuring anything. `run` marks that one path
# safe through git's environment config, which touches no file.
#
# Layout on the VM's disk, so everything survives the reboot a driver swap
# needs:  /data/repo (this repo)  /data/models (HF cache)  /data/results
set -euo pipefail
cmd=${1:?usage: host.sh driver|docker|build|verify|run}; shift || true
IMAGE=specfp8-vllm:0.29.0

case "$cmd" in
driver)
  V=${1:?driver version}
  # Build the module with the compiler that built the running kernel. On
  # GCP's Ubuntu 22.04 the 6.8 kernel is built by gcc-12 while the default
  # cc is gcc-11, which rejects -ftrivial-auto-var-init=zero and fails the
  # build outright. Read the version from /proc/version, never assume it.
  KCC=$(grep -oE 'gcc-[0-9]+' /proc/version | head -1)
  KCC=${KCC:-gcc}
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
       build-essential "$KCC" "linux-headers-$(uname -r)" pkg-config libglvnd-dev
  export CC="$KCC"
  echo "building with $CC ($($CC --version | head -1))"
  printf 'blacklist nouveau\noptions nouveau modeset=0\n' \
       | sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
  curl -fsSLo "/tmp/NVIDIA-$V.run" \
       "https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run"
  # The installer replaces an earlier .run install itself. Installer
  # defaults are kept for every version, so the kernel-module flavour is
  # held fixed across the swap and only the driver version changes.
  sudo CC="$CC" sh "/tmp/NVIDIA-$V.run" --silent --no-questions --ui=none
  echo "installed $V; reboot to load it cleanly, then: host.sh verify"
  ;;
docker)
  curl -fsSL https://get.docker.com | sudo sh
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  ;;
build)
  sudo docker build -t "$IMAGE" /data/repo/cloud/gcp
  ;;
verify)
  echo "== host driver and card =="
  nvidia-smi --query-gpu=driver_version,name,uuid,serial,compute_cap,memory.total \
             --format=csv,noheader
  echo -n "kernel module: "; grep -o 'Open Kernel Module\|NVRM version.*' \
             /proc/driver/nvidia/version | head -1
  echo "== inside the image =="
  sudo docker run --rm --gpus all "$IMAGE" bash -lc '
    python -c "import sys,torch,vllm; print(\"python\",sys.version.split()[0]); \
print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda); \
print(\"vllm\",vllm.__version__); print(\"device\",torch.cuda.get_device_name(0))"
    echo "async flag:"; vllm serve --help=all 2>/dev/null | grep -i -- "async-scheduling" | head -3'
  ;;
run)
  # CPUSET pins the container to those CPUs (e.g. CPUSET=0 for one vCPU),
  # to test F044's CPU candidate. Unset, the container gets every CPU.
  sudo docker run --rm --gpus all --shm-size=16g --ipc=host \
    ${CPUSET:+--cpuset-cpus=$CPUSET} \
    -e SPECFP8_PLATFORM=gcp -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0=/root/spec-fp8-study \
    -v /data/models:/models -v /data/results:/results \
    -v /data/repo:/root/spec-fp8-study \
    "$IMAGE" python cloud/gcp/run_in_container.py "$@"
  ;;
*) echo "unknown: $cmd"; exit 2 ;;
esac
