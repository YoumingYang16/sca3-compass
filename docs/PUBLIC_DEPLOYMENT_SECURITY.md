# Static evidence portal security boundary

The internet-facing product is the static `apps/web/dist-public` build only.
Build with `VITE_PUBLIC_REVIEW_ONLY=true`. Do not expose the local FastAPI,
filesystem, model endpoints, workbench runtime databases or development server.

No patient, login or free-text upload is implemented. There is no application
analytics or tracking script. A hosting provider may still collect ordinary
request/access logs; this is not a guarantee of anonymous infrastructure.
Only allowlisted simulation metrics are in research-evidence.json. This is not
a release of individual clinical data or a certified medical device.

For an HTTPS static host supporting headers, use:

```
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'none'; frame-ancestors 'none'
X-Content-Type-Options: nosniff
Referrer-Policy: no-referrer
X-Frame-Options: DENY
Permissions-Policy: camera=(), microphone=(), geolocation=()
```

Inline styles are used solely for numeric chart widths. No inline scripts or
remote CDN scripts are needed. Hosts without custom response headers need a
separate deployment review; GitHub repository creation does not apply these
headers or publish a website automatically. HTTPS must be verified at the
actual public destination; the loopback preview is not public deployment.

The export scanner looks for dangerous paths, known credential patterns,
authenticated URLs and private-key headers. It is a defense-in-depth check,
not proof that no secret or sensitive material can ever remain. Any uncertain
file stays out of the first upload. No frozen source is edited to mask a hit.

License status: not yet selected by the owner. Do not represent the whole
repository as open source or grant rights to third-party datasets. Keep the
first repository private until rights and sharing scope are settled.
