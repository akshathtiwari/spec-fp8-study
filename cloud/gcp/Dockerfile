# The Modal image, rebuilt for a plain host. Modal's was
#   nvidia/cuda:13.0.3-devel-ubuntu24.04 + add_python("3.11")
#   + apt git + pip vllm==0.29.0 httpx pydantic>=2.0 pyyaml pandas
# torch is left to vLLM's own pin, as there; the records show 2.13.0+cu130
# and the run verifies it before measuring. `modal` is installed only so
# cloud/modal_probe.py imports; nothing in it runs on Modal from here.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
# python-build-standalone 3.11, the same lineage as Modal's add_python;
# --seed provides pip, so the resolver matches Modal's pip_install.
RUN uv python install 3.11 && uv venv --seed /opt/venv --python 3.11
ENV PATH="/opt/venv/bin:${PATH}" VIRTUAL_ENV=/opt/venv
ARG VLLM_VERSION=0.29.0
RUN pip install "vllm==${VLLM_VERSION}" httpx "pydantic>=2.0" pyyaml pandas modal
WORKDIR /root/spec-fp8-study
