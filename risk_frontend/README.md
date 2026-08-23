# Risk assessment frontend

React interface for the Image-Based Traffic Risk Assessment research demo.

## Development

Requires Node.js 22.13 or newer.

```bash
npm ci
npm run dev
```

The Vite development server proxies `/api` to `http://127.0.0.1:8000` by
default. Override the development backend without changing source code:

```bash
VITE_API_PROXY_TARGET=http://127.0.0.1:9000 npm run dev
```

For a separately hosted API, set `VITE_API_BASE_URL` at build time. Leave it
unset for same-origin requests in local development and deployment.

```bash
VITE_API_BASE_URL=https://api.example.org npm run build
```

## Quality checks

```bash
npm run lint
npm test
npm run build
npm audit
```

The interface accepts the v0.1 API contract (`status: ok | uncertain` and
`risk: low | medium | high | unknown`). An `uncertain` response is presented as
an explicit review-required state and must use `risk: unknown`.
