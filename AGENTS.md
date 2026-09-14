# Repository instructions

## Development and release workflow

- Remote: https://github.com/akhastaf/cephalometric-ai.git.
- Start every new development task from the latest `origin/dev` on a new branch. Codex branches use `codex/<descriptive-change>`.
- Never commit implementation changes directly to `dev` or `main`, and never force-push either branch.
- Open implementation pull requests against `dev`. Run Ruff, pytest and the production Docker build first.
- Wait for Abderrazzaq's explicit approval before merging the development PR. After the approved change is merged into `dev`, open a separate `dev` → `main` promotion PR.
- Do not merge a promotion PR, enable auto-merge, configure deployment integrations or deploy without a separate explicit instruction. Deployment is currently paused.
- The one-time initial commit needed to establish this previously empty repository contains only repository policy/CI files; implementation belongs in the first feature PR.
- GitHub plan limitations must be reported accurately: local hooks and CI are not server-enforced branch protection. Do not change visibility, buy a subscription or weaken review settings to bypass a limitation.

## Service boundaries

- Stateless Python/FastAPI, ONNX Runtime CPU; NestJS owns BullMQ, tenant authorization, database access and clinical geometry.
- No production mock/random landmarks, automatic weight downloads, PostgreSQL/Redis clients or permanent bucket credentials.
- Never commit secrets, patient images or unapproved model weights. Preserve model provenance, license and clinical-validation requirements.
- Local integration lives in the parent DentalFlow deployment repository's `docker-compose.dev.yml`. Keep that Compose file and this repository's README consistent.
