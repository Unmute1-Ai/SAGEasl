# Security Policy

## Scope

This repository contains an experimental edge cognition/governance system. Treat autonomous federation, effectors, identity, tool use, and persistence as security-sensitive surfaces.

## Reporting

Do not publish live credentials, device identifiers, private databases, exploit payloads, or user data in public issues. Prefer GitHub private vulnerability reporting/security advisories where available.

## Production boundaries

- Experimental/research status is not a security certification.
- Network/federation features should default off until the deployment trust model is verified.
- Physical effectors must not be enabled solely from model output.
- Secrets and local identity material must remain outside source control.
- Model/tool capability must remain separate from execution authority.
- Persistent memory databases used for tests/research should not be shipped unintentionally.

## Supply chain

Pin release dependencies and model artifacts where practical, record digests, and review submodule provenance before production use.
