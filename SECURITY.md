# Security Policy

## Supported versions

Only the latest release on PyPI is supported with security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately via [GitHub Security Advisories](https://github.com/NineNatthanarong/NRAG/security/advisories/new) — do not open a public issue. You should receive a response within a few days.

## Scope notes

- `nrag serve` (the compilation HTTP service) is a **development tool**: it has no authentication, TLS, or rate limiting. Do not expose it beyond localhost or a trusted network. Hardening it is on the roadmap; reports about its behavior when deployed to the open internet are considered configuration issues, not vulnerabilities.
- Portable bundle import (`nrag import`) validates archive member paths against path traversal. Reports of bypasses are very welcome.
- Index directories and bundles are trusted inputs: only open indexes and import bundles you created or trust.
