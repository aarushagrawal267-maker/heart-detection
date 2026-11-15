# Deployment

QRS Atlas ships as **one Docker image**: the React frontend is pre-built and served by the
FastAPI backend from the same address, and the MIT-BIH dataset is downloaded and verified
against PhysioNet's SHA-256 checksums while the image is built.

> Research and education only. Keep deployments private unless you have a good reason not to.

## Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `QRS_AUTH_PASSWORD` | unset | When set, every page and API call requires HTTP Basic sign-in. `/api/health` stays open for health checks. |
| `QRS_AUTH_USER` | `qrs` | Username for the sign-in prompt |
| `QRS_ALLOW_TRAINING` | `true` | Set `false` to disable the Model Laboratory's training on a shared server |
| `PORT` | `8000` | Port to listen on (Render and Railway set this automatically) |

Resources: about 450 MB RAM while serving; training needs about 1 GB more. The image is about 3 GB (mostly PyTorch).
Experiments trained on a hosted instance are stored on its disk and are lost on redeploy unless
you attach a persistent disk at `/app/data`. The bundled models in `models/` always load.

## Render (recommended of the three)

1. Push this repository to GitHub (already done).
2. Render dashboard → **New → Blueprint** → choose the repository. Render reads `render.yaml`.
3. When asked, enter a value for `QRS_AUTH_PASSWORD`.
4. Wait for the build (about 10 minutes). Open the `onrender.com` URL and sign in as `qrs` with your password.

`render.yaml` uses the `standard` plan (2 GB). The 512 MB `starter` plan is too small.

## Fly.io

```bash
fly launch --copy-config --no-deploy
```

```bash
fly secrets set QRS_AUTH_PASSWORD=choose-a-strong-password
```

```bash
fly deploy
```

`fly.toml` asks for a 2 GB shared-CPU machine and stops it when idle.

## Railway

New Project → Deploy from GitHub repo → this repository. Railway detects `railway.json` and the
Dockerfile. Add the variable `QRS_AUTH_PASSWORD` under Variables, then generate a domain under Settings → Networking.

## Hugging Face Spaces

Add this front matter to the Space's `README.md`, then upload the repository (`sdk: docker`, `app_port: 8000`):

```yaml
---
title: QRS Atlas
sdk: docker
app_port: 8000
---
```

A private Space is already restricted to your Hugging Face account, so `QRS_AUTH_PASSWORD` is optional there.
Docker Spaces need a Hugging Face PRO subscription on free CPU hardware (only static Spaces are free).

## Any server with Docker

```bash
docker build -t qrs-atlas .
```

```bash
docker run -d -p 8000:8000 -e QRS_AUTH_PASSWORD=choose-a-strong-password --name qrs-atlas qrs-atlas
```

Then open `http://your-server:8000`.

## What was verified

The image was built and run locally. It passed:
- the password checks: 401 without or with a wrong password, 200 with the right one, health check open;
- client-side page links and the API docs served from the same address;
- a full-record inference;
- all 9 Playwright end-to-end tests run through the sign-in prompt.

A live deployment on Render, Fly.io, Railway or Hugging Face has not been performed from here.
