# Production Readiness

**Current classification: research/alpha system with runnable core components.**

The package metadata identifies SAGE as an alpha cognition kernel. Production claims should stay aligned with that status until deployment-specific reliability, security, observability, and hardware tests are completed.

## Core gates

- [ ] Clean editable install succeeds on supported Python versions
- [ ] Core test suite passes in CI
- [ ] Package imports/compiles without local machine state
- [ ] Optional GPU/model integrations fail gracefully when unavailable
- [ ] Runtime configuration is documented and reproducible
- [ ] Persistent state/database files are excluded from releases unless intentional

## Edge / hardware gates

- [ ] Jetson image/build instructions are reproduced on target hardware
- [ ] GPU runtime versions are pinned
- [ ] Model artifacts and checksums are versioned
- [ ] Hardware authorization/attestation behavior is tested
- [ ] Sensor and physical-effector integrations remain disabled until separately verified
- [ ] Resource exhaustion / degraded-model behavior is tested

## Governance gates

- [ ] PolicyGate decisions have auditable evidence
- [ ] Model substitution does not silently broaden effector authority
- [ ] Network/federation mode has explicit trust and revocation controls
- [ ] Autonomous/background processes have bounded leases and stop conditions
- [ ] Safety claims are separated from research observations

## Attribution and licensing

Preserve the authorship, license, and upstream project attribution encoded in this repository. Do not relabel upstream research as an Unmute1AI-originated invention. Unmute1AI-specific integrations should be documented as integrations or derivatives where applicable.
