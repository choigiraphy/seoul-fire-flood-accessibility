# Zenodo release checklist

## 1. Connect GitHub repository

1. Sign in to Zenodo.
2. Open GitHub integration.
3. Click `Sync now`.
4. Enable `choigiraphy/seoul-fire-flood-accessibility`.

Official guide:
- https://help.zenodo.org/docs/github/enable-repository/

## 2. Create a GitHub release

1. Open the repository Releases page.
2. Draft a new release.
3. Create a tag such as `v1.0.0`.
4. Publish the release.

GitHub release docs:
- https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository

## 3. Confirm Zenodo archive

After the GitHub release is published, Zenodo should ingest the release automatically and mint:
- a version-specific DOI
- a concept DOI for all future versions

## 4. Update manuscript statements

Replace the placeholder data/code availability wording with the final Zenodo DOI.

Suggested wording:

Data availability: The code and derived analytical outputs supporting this study are available at [ZENODO DOI]. Some underlying source datasets are subject to third-party licensing or redistribution constraints; where redistribution is restricted, the repository provides metadata, provenance, and instructions for re-accessing the original sources.

Code availability: The scripts used to reproduce the main figures and tables are available at [ZENODO DOI].
