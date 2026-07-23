FROM python:3.12.11-slim-bookworm

ARG TARGETARCH
ARG SCIP_VERSION=10.0.2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SCIPOPTDIR=/usr

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    architecture="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${architecture}" in \
      amd64) \
        scip_arch="amd64"; \
        scip_sha256="60eaf196bae6792081c9ac0241315889bee1eada47e043f71a71dac2b3dd0859" \
        ;; \
      arm64|aarch64) \
        scip_arch="aarch64"; \
        scip_sha256="7a8a7f5004217e11cd8f5d26a857c13aeb0325654b6ae925af722cac05491551" \
        ;; \
      *) echo "Unsupported Docker architecture: ${architecture}" >&2; exit 2 ;; \
    esac; \
    url="https://github.com/scipopt/scip/releases/download/v${SCIP_VERSION}/scipoptsuite_${SCIP_VERSION}-1%2Bbookworm_${scip_arch}.deb"; \
    curl --fail --location --retry 3 "${url}" --output /tmp/scip.deb; \
    echo "${scip_sha256}  /tmp/scip.deb" | sha256sum --check -; \
    apt-get update; \
    apt-get install --no-install-recommends -y /tmp/scip.deb; \
    rm -f /tmp/scip.deb; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/scip-cut-trace-v2
COPY . .

RUN python -m pip install --no-cache-dir --upgrade "pip==25.2" \
    && SCIPOPTDIR=/usr python -m pip install --no-cache-dir \
        --no-binary PySCIPOpt -r requirements-core.lock \
    && SCIPOPTDIR=/usr python -m pip install --no-cache-dir \
        --no-build-isolation --no-deps . \
    && python -c "import pyscipopt, scip_cut_trace_v2; model=pyscipopt.Model(); assert (model.getMajorVersion(), model.getMinorVersion(), model.getTechVersion()) == (10, 0, 2)"

CMD ["python", "-m", "unittest", "discover", "-s", "tests"]
