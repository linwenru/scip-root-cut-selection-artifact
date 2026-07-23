# Artifact Release

## Scope and privacy boundary

The working repository remains private because its Git history contains the
manuscript, translations, and reviewer correspondence. Making that repository
public would expose historical revisions even if those files were deleted from
the current tree. Public materials are therefore created only from the
allowlisted release tiers below; `paper/` and internal review notes are excluded
by the archive builder and covered by an automated test.

The publication artifact is split into two content-addressed data archives,
plus a smaller clean public-repository snapshot:

1. `scip-root-cut-selection-artifact-v1.0.0-source.tar.gz` contains source code,
   tests, documentation, and evidence manifests. It is the only input used to
   initialize the history-free public GitHub repository.
2. `scip-root-cut-selection-artifact-v1.0.0-core.tar.gz` contains source code, tests,
   documentation, evidence manifests, active arm results, trained model
   binaries, ranking matrices, and causal JSONL tables.
3. `scip-root-cut-selection-artifact-v1.0.0-observational.tar.gz` contains the compressed
   leakage-safe root observational CSV view.

MIPLIB instance files and the approximately 22 GB raw trace collection are not
redistributed. Instance names and SHA-256 digests are retained in the frozen
plans. The tracer snapshot and fixed split files are included so an authorized
holder of the MIPLIB instances can regenerate the source traces.

## Build

Build the clean public-repository snapshot without manuscript files or private
Git history:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.release_bundle \
  --tiers source manifests \
  --output dist/scip-root-cut-selection-artifact-v1.0.0-source.tar.gz \
  --inventory-output data/manifests/release_public_repository_v1.json
```

Build the core archive and its external inventory from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.release_bundle \
  --output dist/scip-root-cut-selection-artifact-v1.0.0-core.tar.gz \
  --inventory-output data/manifests/release_core_v1.json
```

Build the observational archive separately:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.release_bundle \
  --tiers observational-view \
  --output dist/scip-root-cut-selection-artifact-v1.0.0-observational.tar.gz \
  --inventory-output data/manifests/release_observational_v1.json
```

The tar member metadata and gzip timestamp are normalized. Rebuilding from
unchanged files therefore produces the same archive hash.

## Verify

Each archive embeds `scip-cut-trace-v2/RELEASE_MANIFEST.json`. Verify every
member name, byte count, and SHA-256 digest with:

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.release_bundle \
  --verify dist/scip-root-cut-selection-artifact-v1.0.0-core.tar.gz

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.release_bundle \
  --verify dist/scip-root-cut-selection-artifact-v1.0.0-observational.tar.gz
```

The external inventories additionally record the final archive hash and size.
They are suitable for a repository release page and a DOI deposit record.
Files named `data/manifests/release_*.json` are intentionally excluded from
the archives so an external inventory cannot make its own archive
self-referential or change a subsequent deterministic rebuild.

For an independent closure check, extract the core archive into a clean
directory and run the evidence-index verifier from the extracted root. This
confirms that every public study-claim dependency is present and hash-correct
without access to the private manuscript workspace.

## Container

`Dockerfile` uses Python 3.12 on Debian Bookworm and installs the official SCIP
10.0.2 Bookworm package for either amd64 or arm64. Both architecture-specific
packages are checked against the SHA-256 digests published with the official
SCIP release. PySCIPOpt 6.2.1 is then built against that installation.

```bash
docker build --tag scip-cut-trace-v2:0.1.0 .
docker run --rm scip-cut-trace-v2:0.1.0
```

The default container command runs the complete unit-test suite. MPS files and
large trace inputs should be mounted at run time rather than copied into the
image.

## Publication Checklist

The following research artifacts are ready:

- frozen machine-readable experiment plans;
- 480 complete active arm records for 40 instances and three seeds;
- 10,000-replicate cluster-bootstrap statistics;
- multi-instance no-op and direct-hybrid parity records;
- a frozen 37-feature XGBoost ranker and its 162-record, 18-Group independent
  complete-solve No-Go pilot;
- deterministic archive builder, inventories, and container recipe;
- exact public evidence dependency snapshots;
- final author metadata in `CITATION.cff` and Zenodo deposit metadata;
- public repository URL:
  <https://github.com/linwenru/scip-root-cut-selection-artifact>;
- Apache-2.0 license and third-party provenance notice.

The following items require author action before a public release:

- confirm the remaining conflict-of-interest and generative-AI declarations;
- create the history-free public repository from the allowlisted snapshot;
- tag the immutable release after all checks pass;
- upload both archives to a long-term repository such as Zenodo;
- insert the assigned DOI into the manuscript, README, and `CITATION.cff`;
- confirm that the selected MIPLIB redistribution statement matches the
  repository terms in force at publication time.

No DOI or immutable archival-release claim is asserted before those steps are
completed.
