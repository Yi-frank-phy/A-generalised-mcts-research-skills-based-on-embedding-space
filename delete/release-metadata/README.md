# Release metadata archive

This folder contains superseded generated release metadata for comparison only.

`bundle-manifest.old.json` is the stale manifest that caused CI to stop before pytest after the production runtime changed. The live release manifest is regenerated from the current production tree by CI, then verified and copied into the managed-template bundle before package validation.
