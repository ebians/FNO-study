# README Sync Checklist (EN/JA)

This file defines a simple operating rule to keep `README.md` and `README_ja.md` synchronized.

## Rule

1. Update both files in the same commit whenever one README is changed.
2. Keep section order aligned between English and Japanese versions.
3. Keep command examples functionally equivalent.
4. Record the same sync version in both files under the sync policy section.

## Quick Check Before Commit

- [ ] `README.md` and `README_ja.md` were both reviewed.
- [ ] The following sections exist in both files:
  - Quick Start
  - Next Step: Training Dataset
  - Train FNO
  - Inference (Single Case)
  - README Sync Policy
- [ ] PowerShell command examples are updated in both files.
- [ ] Newly added scripts/files are documented in both files.
- [ ] Sync version string matches in both files.

## Sync Version Format

Use this format in both README files:

`Sync Version: YYYY-MM-DD`
