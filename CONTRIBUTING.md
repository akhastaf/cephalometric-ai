# Development and promotion

`dev` integrates reviewed changes. `main` receives only separately reviewed promotions from `dev`. Deployment is paused; this repository has no deployment workflow.

Start each task from a new branch:

```sh
git fetch origin
git switch -c codex/describe-the-change origin/dev
git config core.hooksPath .githooks
```

Commit only the task changes and run:

```sh
docker build -t dentalflow-ceph-ai:test .
docker build -f Dockerfile.test -t dentalflow-ceph-ai-tests:test .
docker run --rm dentalflow-ceph-ai-tests:test ruff check app tests
docker run --rm dentalflow-ceph-ai-tests:test python -m pytest -q
git push -u origin HEAD
gh pr create --base dev
```

Wait for Abderrazzaq's explicit approval. Once the approved PR is merged into `dev`, create the promotion PR with `gh pr create --base main --head dev`. Do not merge it or deploy until separately authorized. Use a merge commit for a `dev` → `main` promotion to preserve shared history; do not delete `dev` afterward.

## Protection configuration

The intended protection for both branches is in `.github/branch-protection.json`: pull requests, one approving code-owner review, stale approval dismissal, required tests/build/branch-policy checks, resolved conversations, enforcement for admins, and no force pushes or deletion. `.github/CODEOWNERS` assigns review to `@akhastaf`.

GitHub currently rejects branch protection for this private repository with HTTP 403 requiring GitHub Pro. These settings are **not enforced** until the owner enables a supported plan and the API accepts the configuration. Local `.githooks` protect against accidental direct commits/pushes in configured clones, and CI rejects a promotion to `main` from any source other than this repository's `dev`; neither replaces GitHub protection.

After the plan supports protection, apply and read back both branches:

```sh
for branch in main dev; do
  gh api --method PUT "repos/akhastaf/cephalometric-ai/branches/$branch/protection" --input .github/branch-protection.json
  gh api "repos/akhastaf/cephalometric-ai/branches/$branch/protection"
done
```

GitHub does not allow PR authors to approve their own PRs. A PR opened using `akhastaf` credentials needs another approved development identity to author it if `akhastaf` is to supply the required code-owner review. Do not disable required review to work around this. Conversational approval is an instruction to the coding agent, not a GitHub review event.

References: [protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches), [required reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/approving-a-pull-request-with-required-reviews).
