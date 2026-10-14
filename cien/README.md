# CIEŃ public connection metadata — temporary channel

This directory contains only a signed public connection manifest for the account owner's CIEŃ test deployment. GitHub is an interim outage/discovery channel, not the intended long-term distribution infrastructure and not the CIEŃ source repository. No user records, diagnostics, private keys or application data belong here. Calls and user messages do not pass through GitHub.

`connection-v1.json` is an envelope of a base64 UTF-8 payload and an ECDSA P-256/SHA-256 signature over `CIEN-CONNECTION-V1\n` followed by the exact payload bytes. Android verifies a public key shipped in its signed APK, a fixed service identifier, expiration, increasing version and payload digest. This file cannot replace the trust key. API endpoints retain platform TLS, hostname and SPKI validation. Both current API tunnels use Cloudflare and the same local server; this GitHub document is a separate information/discovery channel, not an independent replica of the service.

Only signed public metadata is published. A signature provides authenticity and integrity, not confidentiality. The file publicly associates the endpoint addresses with this GitHub account; this is not an anonymity mechanism. Updates can be cached by GitHub, and Android may delay background checks.

The future design should support replaceable configuration sources across independent providers, preserving the APK trust anchor and rollback protection while separately addressing metadata privacy and availability. The current APK pins one GitHub source URL; changing that source currently requires an APK update. No domain or hosting purchase, Tor service, or automatic remote trust-key replacement is part of this interim deployment.
