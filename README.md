# Dajoong

Dajoong is the production drawing-to-BIM product repository. It contains the
CPU conversion service and the shared web, Android, and iOS Studio clients.

## Repository boundaries

- `modules/plan2bim` contains the server-side drawing-to-IFC/GLB compiler.
- `apps/plan2bim-studio/backend` exposes authenticated conversion jobs and artifacts.
- `apps/plan2bim-studio/frontend` is the shared web and Capacitor mobile client.
- `apps/plan2bim-studio/infra` contains the private AWS runtime and Cloudflare boundaries.
- `docs/COMMERCIAL_RELEASE.md` lists the keys required for a commercial release.

The browser and mobile bundles never contain ONNX checkpoints, model parameters,
server credentials, or an inference runtime. All conversions run on Dajoong-owned
AWS services through `https://studio-api.dajoongbim.com`. The public
Cloudflare API proxy signs origin requests; direct App Runner job access is denied.

## Private model delivery

Model checkpoints are deliberately absent from Git. A release workflow downloads
an encrypted, access-controlled bundle from Dajoong-owned object storage, verifies
its SHA-256, safely extracts it, and only then builds the private server image.
See `docs/PRIVATE_MODEL_DELIVERY.md`.

## Local development

Install the private model bundle first, then:

```powershell
py -3.12 -m pip install -e modules/plan2bim -e "apps/plan2bim-studio/backend[dev,aws,auth]"
python -m pytest modules/plan2bim/tests apps/plan2bim-studio/backend/tests -q

cd apps/plan2bim-studio/frontend
npm ci
npm test
npm run dev
```

No production credential belongs in this repository. Configure GitHub's protected
`production` environment and use the deployment workflows described in the release guide.
