# Connect to an existing Ignite service

This path creates only a personal connection. Obtain the namespace, Service
name, served model ID and port from its owner. Obtain your own Ignite kubeconfig
from Rancher and verify its context and API server there. The helper requires
the exact `--expected-server` value, including an explicit `:443` if present.
The service must support the OpenAI-compatible `/v1/models` and chat APIs.

Use Python 3.10+, kubectl (the pilot used 1.35.6) and OpenSSH on PATH. The helper
is standard-library Python and works from Linux or PowerShell; on Windows use
`python` if `python3` is not installed. It does not install tools or change the
original kubeconfig. The caller needs get on Services, list/get on pods and
create on pods/portforward in the selected namespace.

## From Lux/Linux or an AMD workstation

Run from the skill directory, substituting the values verified for your account:

```bash
python3 scripts/connect.py --kubeconfig /path/to/your/ignite.yaml \
  --context YOUR_IGNITE_CONTEXT --expected-server YOUR_VERIFIED_HTTPS_SERVER \
  --namespace YOUR_NAMESPACE --service YOUR_SERVICE --model le-chonk \
  --ssh-host YOUR_OLCF_USER@home.ccs.ornl.gov --test
```

Enter your PIN/SecurID in the SSH terminal prompt. Do not give it to the agent.
The helper opens a private SSH SOCKS connection, writes a temporary routed
kubeconfig accessible only to your account, then starts localhost port forwarding.
Use the identical arguments on one line in PowerShell. `--kubectl` accepts a full
executable path. A configured SSH alias is also accepted by `--ssh-host`.

If you already have your own SOCKS5 tunnel, substitute `--socks-port 10000` for
`--ssh-host`. On a host with direct Rancher API access, normally
`home.ccs.ornl.gov`, use `--direct`. These three routing options are mutually
exclusive. The direct mode removes proxy settings only in its temporary config.
The [OLCF access guide](https://docs.olcf.ornl.gov/systems/lux_user_guide.html#navigating-to-the-rancher-dashboard)
documents the gateway route. Do not assume a Kubernetes ClusterIP is reachable
from a Lux Slurm login node or workstation.

The default local endpoint is `http://127.0.0.1:18000/v1`. `--port` selects another
unused local port; `--service-port` changes the target Service port from 8000.
The helper refuses an occupied port, checks the expected model ID and never
binds an AMD-wide or public listener. It stops on authentication/TLS failures.

Keep the terminal open and send API requests from the **same host**:

```bash
curl --noproxy '*' http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"le-chonk","messages":[{"role":"user","content":"What is 17 times 23? Give the final answer."}],"max_tokens":4096,"reasoning_effort":"low","temperature":0}'
```

Require a nonempty answer, here 391. ML4's reasoning and answer share the output
token budget. Its `reasoning_effort` parameter is top-level. Do not copy the
different Muse/Inkling parameter conventions from the Slurm skill.

The tested Service has no separate inference key through this personal tunnel;
clients requiring a key field can use `unused`. Authentication is at SSH and
Kubernetes. Never use a Rancher token as an inference API key. An API-key-protected
gateway is outside this helper's current scope. ClusterIP alone supplies no API
authentication, so wider network sharing requires site-approved access controls.

`--check --test` verifies one connection/completion and exits. Otherwise the
helper reconnects a lost port-forward until interrupted; it does not resubmit
in-flight inference. Ctrl+C closes only processes started by this invocation,
removes its temporary config, and leaves the model and any reused SOCKS tunnel
running. A forced kill may leave its user-private temporary directory behind.
Refresh expiring kubeconfigs through Rancher; the helper does not renew them.
