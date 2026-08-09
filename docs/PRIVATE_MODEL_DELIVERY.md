# Private model delivery

The production checkpoint is not stored in Git, Git LFS, a release attachment, a
mobile application, or a browser bundle.

Create a gzip-compressed tar archive whose files are either at the archive root or
inside one `models/` directory. It must contain each `.onnx` checkpoint and its
matching `.onnx.json` content-addressed manifest. Qualification and calibration JSON
files may be included in the same bundle.

Upload it to a private, versioned, encrypted S3 bucket. Give only the GitHub deployment
OIDC role read access to that exact object. Set:

- GitHub environment variable `DAJOONG_MODEL_BUNDLE_S3_URI`
- GitHub environment secret `DAJOONG_MODEL_BUNDLE_SHA256`

The deployment workflow downloads the object after obtaining a short-lived AWS role,
verifies the archive hash and member paths, installs it into the server package, and
builds a private ECR image. The image is never published to a public registry.

For local development, download the same bundle with an authorized AWS profile and run:

```powershell
python scripts/install_private_models.py private-models.tar.gz `
  modules/plan2bim/src/buili_plan2bim/models `
  --sha256 <64-character-sha256>
```
