# syntax=docker/dockerfile:1.7
# The miles image: upstream miles' reference image, moved onto the Databricks AI Runtime base.
#
#   bash frameworks/miles/image/build.sh <registry>/<repository>:<tag>
#
# The reference is the image miles' CI builds for a release (CUDA 13.0, PyTorch 2.14, SGLang's
# sglang-miles branch, Megatron-LM's miles-main). It is 41 GB, twice AI Runtime's 20 GB image limit
# (which counts the unpacked size), so this image takes the reference's Python environment as built
# and leaves out:
#   - FlashInfer's prebuilt kernels for every GPU architecture (8.7 GB): FlashInfer compiles the
#     kernels a run uses on first use, which is why the image keeps nvcc and a C++ compiler;
#   - FlashAttention 3 (1.5 GB), which runs on Hopper only: on H100 the cookbook's launchers serve
#     with SGLang's Triton attention instead (rft_miles/launchers), as on B300;
#   - packages nothing on the cookbook's path imports (2.3 GB): NVIDIA Dynamo and the JAX stack
#     only its simulator uses, NIXL (prefill/decode disaggregation), MSCCL++ (which needs GDRCopy,
#     absent from the AI Runtime base), Mamba's kernels, SGLang's opt-in TokenSpeed MLA and
#     HPC-Ops backends, and the diffusion guardrail stack;
#   - static libraries (nothing links them at run time) and compiled bytecode;
#   - miles itself: the cookbook installs the miles commit in rft_miles/miles.lock.json at job start.
#     The build checks that the reference was built from that same commit.
# Nothing else changes: no patches and no package swaps.
ARG REFERENCE_IMAGE=radixark/miles@sha256:cb2a5f35a7de0915cb05fbdb0d2e7a7be0b495eb35c71ef3f10192401bcf18b0
ARG AIR_BASE=databricksruntime/air@sha256:8da6cc766e745239ed46f573344a6cad351deb2c0caf3135b26e98f20e9197c0

FROM ${REFERENCE_IMAGE} AS build
ARG REFERENCE_IMAGE
ARG MILES_COMMIT
# The per-architecture caches share one package directory, which their uninstall leaves behind:
# FlashInfer would take it for an installed cache.
RUN test "$(git -C /root/miles rev-parse HEAD)" = "${MILES_COMMIT}" && \
    python -m pip uninstall -y miles flash-attn-3 flashinfer-cubin flashinfer-jit-cache \
      flashinfer-jit-cache-sm80 flashinfer-jit-cache-sm89 flashinfer-jit-cache-sm90a \
      flashinfer-jit-cache-sm100a flashinfer-jit-cache-sm103a flashinfer-jit-cache-sm120f \
      ai-dynamo ai-dynamo-runtime aisimulate bokeh plotly \
      jax jaxlib flax optax chex orbax-checkpoint jaxopt equinox tensorstore \
      nixl nixl-cu13 mscclpp mamba-ssm tokenspeed-mla tokenspeed-triton hpc-ops \
      cosmos-guardrail moviepy imageio-ffmpeg tfp-nightly && \
    rm -rf /opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache && \
    rm -f /opt/sglang/lib/python3.12/site-packages/triton/backends/nvidia/lib/cupti*/*.a && \
    find /opt/sglang /opt/python /sgl-workspace/sglang/python /root/Megatron-LM/megatron /root/torchtitan/torchtitan \
      -name __pycache__ -type d -prune -exec rm -rf {} + && \
    printf '{"reference": "%s", "miles": "%s", "sglang": "%s", "megatron": "%s", "torchtitan": "%s"}\n' \
      "${REFERENCE_IMAGE}" "${MILES_COMMIT}" \
      "$(git -C /sgl-workspace/sglang rev-parse HEAD)" \
      "$(git -C /root/Megatron-LM rev-parse HEAD)" \
      "$(git -C /root/torchtitan rev-parse HEAD)" > /opt/sglang/stack.json

FROM ${AIR_BASE} AS runtime

# AI Runtime provides the CUDA runtime and the EFA/NCCL network stack. FlashInfer, TileLang,
# DeepGEMM and Triton compile kernels at run time: nvcc, the CUDA headers and a C++ compiler. Of
# the static libraries, they need only the CUDA runtime's, which nvcc links by default.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      cuda-nvcc-13-0=13.0.88-1 \
      cuda-cudart-dev-13-0=13.0.96-1 \
      libcublas-dev-13-0=13.1.1.3-1 \
      cuda-nvrtc-dev-13-0=13.0.88-1 \
      build-essential=12.10ubuntu1 \
      ninja-build=1.11.1-2 && \
    find /usr/local/cuda-13.0/targets/x86_64-linux/lib -name '*.a' \
      ! -name libcudart_static.a ! -name libcudadevrt.a -delete && \
    rm -rf /var/lib/apt/lists/*

# The reference's interpreter and virtual environment, at AI Runtime's /opt/venv, and the
# source trees its editable installs (SGLang, Megatron-LM, TorchTitan) point into.
COPY --from=build /opt/python/ /opt/python/
COPY --from=build /opt/sglang/ /opt/venv/
COPY --from=build /sgl-workspace/sglang/python/ /sgl-workspace/sglang/python/
COPY --from=build /root/Megatron-LM/megatron/ /root/Megatron-LM/megatron/
COPY --from=build /root/torchtitan/torchtitan/ /root/torchtitan/torchtitan/
COPY image/verify_image.py /opt/venv/verify_image.py

ENV CUDA_HOME=/usr/local/cuda
# FlashInfer's sampling kernels include cuRAND, whose headers come with the CUDA wheels.
RUN grep -lI '^#!/opt/sglang/bin/python' /opt/venv/bin/* | xargs sed -i '1s|^#!/opt/sglang/|#!/opt/venv/|' && \
    sed -i 's|/opt/sglang|/opt/venv|g' /opt/venv/bin/activate* && \
    ln -s /opt/venv/lib/python3.12/site-packages/nvidia/cu13/include/curand*.h /usr/local/cuda/include/

# OfficeQA's tools and judge, as in the verl image: BM25 retrieval (bm25s), the workspace AI
# gateway's credentials (databricks-sdk) and the strict compute sandbox (bubblewrap). Everything
# they import is already here, so --no-deps leaves the reference's packages as they are
# (databricks-sdk's own requirements would move protobuf 7 back to 6). PIP_INDEX_URL, if set,
# reaches the build as the BuildKit secret `pip_index` (build.sh), never as a build argument.
RUN --mount=type=secret,id=pip_index \
    apt-get update && apt-get install -y --no-install-recommends bubblewrap=0.9.0-1ubuntu0.3 && \
    rm -rf /var/lib/apt/lists/* && bwrap --version && \
    INDEX="$(cat /run/secrets/pip_index 2>/dev/null || true)" && \
    python -m pip install --no-cache-dir --no-deps ${INDEX:+--index-url "${INDEX}"} \
      databricks-sdk==0.146.0 bm25s==0.2.14

# The build machine has no GPU driver: the check loads the CUDA driver stub, which the image does
# not keep (on AI Runtime the driver provides libcuda).
RUN --mount=type=bind,from=build,source=/usr/local/cuda/lib64/stubs/libcuda.so,target=/run/cuda-stub/libcuda.so.1 \
    cd /tmp && LD_LIBRARY_PATH=/run/cuda-stub:${LD_LIBRARY_PATH} PYTHONDONTWRITEBYTECODE=1 python /opt/venv/verify_image.py
