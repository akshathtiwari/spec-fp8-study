#!/usr/bin/env bash
# Host-side steps for F042's driver-swap intervention, run ON the GCP VM.
# Each subcommand is one step, run separately so each can be verified before
# the next spends money.
#
#   host.sh driver 580.95.05   install (or replace) the NVIDIA driver
#   host.sh docker             Docker + NVIDIA Container Toolkit
#   host.sh build              the Modal-equivalent image (cloud/gcp/Dockerfile)
#   host.sh verify             driver, card, torch, vllm, async flag -- no GPU load
#   host.sh gvisor 20260914.0  install that upstream runsc as Docker runtime "runsc"
#   host.sh run ARGS...        cloud/gcp/run_in_container.py ARGS
#                              (CPUSET=0 host.sh run ... pins to one vCPU)
#                              (RUNTIME=runsc ... runs it under gVisor nvproxy)
#                              (THREADS1=1 ... library thread defaults of one)
#   VLLM=0.30.0 host.sh build|verify|run   another engine release
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
VLLM=${VLLM:-0.29.0}
IMAGE=specfp8-vllm:$VLLM

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
  sudo docker build --build-arg VLLM_VERSION="$VLLM" -t "$IMAGE" /data/repo/cloud/gcp
  ;;
gvisor)
  # Modal (support reply, 2026-11-03): GPU Functions run under gVisor with
  # nvproxy; the 2026-10-04 containers used their build "Release20260916"
  # with driver 580.95.05 and -nvproxy-allow-unsupported-driver. Upstream
  # release-20260914.0 was published 2026-09-16 and registers 580.95.05 as
  # an UNSUPPORTED ABI derived from 580.65.06, usable only with that flag.
  # Modal's build is their own (cfce95d57233 is not an upstream commit), so
  # this is the closest public build, not a copy of theirs.
  R=${1:?gVisor release, e.g. 20260914.0}
  B="https://storage.googleapis.com/gvisor/releases/release/$R/x86_64"
  sudo apt-get install -y -qq zstd
  rm -rf /tmp/gv && mkdir -p /tmp/gv && cd /tmp/gv
  curl -fsSLO "$B/gvisor.tar.zstd" && curl -fsSLO "$B/gvisor.tar.zstd.sha512"
  sha512sum -c gvisor.tar.zstd.sha512
  tar --zstd -xf gvisor.tar.zstd
  RUNSC=$(find /tmp/gv -type f -name runsc | head -1)
  sudo install -m 755 "$RUNSC" "/usr/local/bin/runsc-$R"
  # Since 2026 runsc starts the sentry as a sidecar it looks up in
  # gvisor-bin/ beside itself, and refuses to run without it (STRICT).
  sudo rm -rf /usr/local/bin/gvisor-bin
  sudo cp -r "$(dirname "$RUNSC")/gvisor-bin" /usr/local/bin/gvisor-bin
  "/usr/local/bin/runsc-$R" --version
  sudo python3 - "$R" <<'EOF'
import json, sys, pathlib
p = pathlib.Path("/etc/docker/daemon.json")
d = json.loads(p.read_text()) if p.exists() else {}
d.setdefault("runtimes", {})["runsc"] = {
    "path": f"/usr/local/bin/runsc-{sys.argv[1]}",
    "runtimeArgs": ["--nvproxy-allow-unsupported-driver",
                    "--debug-log=/data/runsc/", "--debug"]}
p.write_text(json.dumps(d, indent=2))
print(json.dumps(d["runtimes"]["runsc"]))
EOF
  sudo mkdir -p /data/runsc && sudo systemctl restart docker
  echo "== gVisor sees the GPU? =="
  sudo docker run --rm --runtime=runsc --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 \
       bash -c 'dmesg | head -2; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader'
  echo "== nvproxy's own verdict on the driver =="
  sudo grep -rh "not officially supported\|unsupported Nvidia" /data/runsc/ | head -2
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
  # RUNTIME=runsc: gVisor has no host IPC namespace to share, and Modal's
  # containers do not get one either, so --ipc=host is dropped there.
  # THREADS1=1: Modal sets library thread-count defaults to one (support
  # reply, 2026-11-03), which is why no Modal log shows torch reducing its
  # threads. Both are recorded on every measurement.
  RT=${RUNTIME:-runc}
  SANDBOX=$RT
  [ "$RT" = runsc ] && SANDBOX=$(grep -o 'runsc-[0-9.]*[0-9]' /etc/docker/daemon.json | head -1)
  sudo docker run --rm --runtime="$RT" --gpus all --shm-size=16g \
    $([ "$RT" = runc ] && echo --ipc=host) \
    ${CPUSET:+--cpuset-cpus=$CPUSET} \
    ${THREADS1:+-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1} \
    -e SPECFP8_SANDBOX="$SANDBOX" \
    -e SPECFP8_PLATFORM=gcp -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0=/root/spec-fp8-study \
    -v /data/models:/models -v /data/results:/results \
    -v /data/repo:/root/spec-fp8-study \
    "$IMAGE" python cloud/gcp/run_in_container.py "$@"
  ;;
*) echo "unknown: $cmd"; exit 2 ;;
esac
