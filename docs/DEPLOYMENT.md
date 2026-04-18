# Deployment Guide

Operators' reference — read this when something's broken or you're setting up a new environment.

---

## 1. Personal machine (Anthropic direct)

```bash
export BUNDLE_PLATFORM_LLM=anthropic
export ANTHROPIC_API_KEY=<your key>
uv sync
```

---

## 2. Work laptop via AWS Bedrock

```bash
export BUNDLE_PLATFORM_LLM=bedrock
uv sync --extra bedrock
```

**Credentials (pick one):**

- IAM role (preferred) — no extra env vars needed; role must have `bedrock:InvokeModel` permission.
- Static keys:
  ```bash
  export AWS_ACCESS_KEY_ID=<key>
  export AWS_SECRET_ACCESS_KEY=<secret>
  export AWS_REGION=us-east-1
  ```

---

## 3. Work laptop via GCP Vertex AI

```bash
export BUNDLE_PLATFORM_LLM=vertex
export VERTEX_PROJECT=<gcp-project-id>
export VERTEX_REGION=us-central1
uv sync --extra vertex
gcloud auth application-default login
```

---

## 4. Work laptop via Azure AI Foundry

```bash
export BUNDLE_PLATFORM_LLM=azure
export AZURE_ENDPOINT=https://<your-endpoint>.openai.azure.com/
export AZURE_API_KEY=<your key>
uv sync
```

No extra dependency group needed.

---

## 5. Corporate proxy / TLS inspection

If your network performs TLS inspection, point the CA bundle at the corporate certificate:

```bash
export REQUESTS_CA_BUNDLE=/path/to/corporate-ca.pem
export SSL_CERT_FILE=/path/to/corporate-ca.pem
```

If running sentence-transformers in an air-gapped environment:

```bash
export HF_HUB_OFFLINE=1
```

The model must already be cached locally (typically under `~/.cache/huggingface/`).

---

## 6. Data-egress warning

Bundle chunks contain log excerpts from the indexed support bundle. When you run evals with `--ragas`, those chunks are sent to the configured LLM API.

- **Before running on production bundle data**, confirm with your security team.
- **To keep data in your AWS region**: use `BUNDLE_PLATFORM_LLM=bedrock`.
- **To skip LLM calls entirely**: omit `--ragas`. Deterministic metrics (recall, precision, MRR) run with zero API cost.

---

## 7. Running an eval scorecard end-to-end

```bash
# 1. Index a bundle with log-analyse (separate repo)
cd ~/log-analyse
uv run log-analyse --run-id esxi-real-001
uv run log-analyse-load-qdrant --run-id esxi-real-001

# 2. Verify the contract
cd ~/bundle-platform
QDRANT_URL=http://localhost:6333 \
BUNDLE_PLATFORM_COLLECTION=support_bundle_chunks \
BUNDLE_PLATFORM_EMBED_MODEL=BAAI/bge-small-en-v1.5 \
uv run pytest tests/integration/test_contract_smoke.py -v

# 3. Run deterministic eval (zero API cost)
uv run bundle-platform-eval \
  --collection support_bundle_chunks \
  --golden eval/golden/esxi/ \
  --strategies baseline,with_dedup,with_drain3,with_rerank,combined \
  --no-ragas

# 4. Run with RAGAS (costs API tokens — only after deterministic looks good)
ANTHROPIC_API_KEY=<key> \
uv run bundle-platform-eval \
  --collection support_bundle_chunks \
  --golden eval/golden/esxi/ \
  --strategies baseline,with_rerank \
  --ragas
```
