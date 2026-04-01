# AGENTS.md

This document provides a guide for agents working on YugabyteDB

### Deploying and running

For agents that want to deploy, configure and run YugabyteDB refer to instructions at ./docs/content/stable/quick-start

### Repo Structure

| Directory | What it contains |
|---|---|
| `src/` | Core database code: PostgreSQL fork (`src/postgres/`), YugabyteDB C++ storage engine (`src/yb/`), Odyssey connection pooler (`src/odyssey/`) |
| `java/` | Java client library, CDC connector, and DB tests |
| `managed/` | YugabyteDB Anywhere (YBA) platform — orchestration UI, CLI, node agent, and backend (Scala/Java) |
| `docs/` | Source files for the docs website (docs.yugabyte.com) |
| `python/` | Python build utilities and test infrastructure scripts |
| `build-support/` | Build system scripts, linting, and third-party dependency tooling |
| `cmake_modules/` | CMake modules for locating dependencies and custom build functions |
| `cloud/` | Docker, Kubernetes, and Grafana deployment configurations |
| `yugabyted-ui/` | Yugabyted web UI (React frontend + Go API server) |
| `architecture/` | Internal design documents and architecture specs |
| `troubleshoot/` | Troubleshooting framework backend and UI |

### Coding and Development

When working on DB code (`src/`), refer to `src/AGENTS.md` for build and test guidance

## Cursor Cloud specific instructions

### Running YugabyteDB

The core database (C++ build via `yb_build.sh`) takes hours to compile from source. Use the official Docker image instead:

```bash
sudo dockerd &>/tmp/dockerd.log &
sleep 3
sudo chmod 666 /var/run/docker.sock
docker run -d --name yugabyte \
  -p 7000:7000 -p 9000:9000 -p 15433:15433 -p 5433:5433 -p 9042:9042 \
  yugabytedb/yugabyte:latest bin/yugabyted start --background=false
```

Wait ~30s for startup, then verify: `docker exec yugabyte yugabyted status`

Connect via YSQL: `docker exec -it yugabyte bin/ysqlsh -h $(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' yugabyte) -U yugabyte -d yugabyte`

Key ports: YSQL=5433, YCQL=9042, YB-Master UI=7000, YB-TServer UI=9000, Yugabyted UI=15433.

### Development tooling

| Component | Build/Run | Lint/Test |
|---|---|---|
| `java/` (Maven) | `cd java && mvn install -pl interface-annotations -DskipTests` | `mvn validate`. Note: `yb-client` and most modules require protoc from the C++ third-party build. |
| `yugabyted-ui/ui` (React/Vite) | `cd yugabyted-ui/ui && npm install && npm run build` | `npx tsc --noEmit` (pre-existing TS errors exist) |
| `yugabyted-ui/apiserver` (Go 1.24) | Build UI first, copy to `apiserver/cmd/server/ui`, then `go build ./cmd/server/` | `go vet ./...` |
| `managed/ui` (React/Vite) | `cd managed/ui && npm install` | `npx eslint --ext .js,.jsx,.ts,.tsx src/`. `npm run build` requires generated v2 API client (needs sbt backend). Dev server works: `npx vite` |
| `troubleshoot/troubleshooting-framework-ui` | `cd troubleshoot/troubleshooting-framework-ui && npm install --legacy-peer-deps` | Standard npm scripts |
| Python utilities | `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` | `python -m pycodestyle` |

### Environment notes

- Node.js >= 22 is required (use nvm). Go 1.24 is required for `yugabyted-ui/apiserver`.
- The `managed/ui` production build fails without a generated v2 API client (`openapi.yaml` from the sbt backend). The dev server (`npx vite`) starts fine for iterative development.
- The `troubleshoot/troubleshooting-framework-ui` requires `npm install --legacy-peer-deps` due to peer dependency conflicts.
- Docker is needed for running YugabyteDB; Docker-in-Docker requires `fuse-overlayfs` storage driver and `iptables-legacy` in Cloud Agent VMs.
- The `.cursor/environment.json` references a Dockerfile pointing to `yugabyteci/yb_build_infra_almalinux9_x86_64:latest` (the C++ build image); this is not used for Cloud Agent setups since we install tools directly.
