# Release & Rollback Workflow

## Versioning
- Use [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH)
- Tag releases: `git tag -a v1.2.3 -m "Release v1.2.3" && git push --tags`

## Branching
- `main`: production
- `staging`: pre-release
- `dev`: daily work

## Release Steps
1. Bump version, update CHANGELOG
2. Commit: `git commit -am "release: vX.Y.Z"`
3. Tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. Push: `git push origin main --tags`
5. Deploy from tag or build Docker image

## Rollback
- `git checkout vX.Y.Z` (previous tag)
- Redeploy from this tag or Docker image

## CI/CD
- All PRs must pass CI (lint, Streamlit smoke test)
- Use feature flags for risky features

## Dependency Management
- Edit `requirements.in`, run `pip-compile` to update `requirements.txt`
- Use `pip-sync` to install exact versions

## Secrets
- Never commit `.streamlit/secrets.toml` or any API keys
