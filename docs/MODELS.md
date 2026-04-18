# Model IDs by Backend

| Backend | Sonnet | Opus |
|---|---|---|
| Anthropic direct | claude-sonnet-4-6 | claude-opus-4-7 |
| AWS Bedrock | anthropic.claude-sonnet-4-5-20251001-v1:0 | anthropic.claude-opus-4-7-20260215-v1:0 |
| GCP Vertex | claude-sonnet-4-5@20251001 | claude-opus-4-7@20260215 |
| Azure AI Foundry | claude-sonnet-4-6 | claude-opus-4-7 |

Set `BUNDLE_PLATFORM_MODEL` to override the default for a given backend.
