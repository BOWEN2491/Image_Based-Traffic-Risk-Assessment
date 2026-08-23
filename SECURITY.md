# Security Policy

## Supported versions

Security fixes are provided for the latest tagged release and the current `main` branch. Earlier research snapshots are unsupported.

## Reporting a vulnerability

Please do not open a public issue for an unpatched vulnerability. Use GitHub's **Report a vulnerability** private advisory form in the repository Security tab. Include affected versions, reproduction steps, impact, and any suggested mitigation. Do not include private data or third-party credentials.

You should receive acknowledgement within seven days. The maintainer will investigate, coordinate a fix and disclosure date where practical, and credit reporters who wish to be named. This is a personal research project, so response times are best effort and no security SLA is offered.

## Security boundaries

The API accepts untrusted image uploads, but it is a local research demo rather than an internet-hardened multi-tenant service. Put authentication, TLS, request-rate controls, process isolation, and resource monitoring in front of it before any network exposure.

Model assets are accepted only when their SHA-256 and size match the release manifest. Do not replace native/weights-only formats with untrusted Python pickle or joblib files.

## Safety issues

Incorrect risk estimates are model-safety issues rather than software-security vulnerabilities. They are still welcome as public issues when they contain no sensitive exploit details. This software must not be used for safety-critical decisions.
