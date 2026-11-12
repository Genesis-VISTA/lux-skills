#!/usr/bin/env python3
"""Connect to the existing Ignite service using the caller's own credentials."""
import argparse
import csv
import json
import os
from pathlib import Path
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

API_HOST = 'console.apps.slate-mod.ccs.ornl.gov'


def executable(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f'{name} was not found. See references/connect.md for prerequisites.')
    return found


def free_port(port=0):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
        return sock.getsockname()[1]


def check_socks(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=2) as sock:
            sock.settimeout(2)
            sock.sendall(b'\x05\x01\x00')
            reply = b''
            while len(reply) < 2:
                part = sock.recv(2 - len(reply))
                if not part:
                    break
                reply += part
            return reply == b'\x05\x00'
    except OSError:
        return False


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def private_directory():
    folder = Path(tempfile.mkdtemp(prefix='le-chonk-')).resolve()
    try:
        if os.name == 'nt':
            identity = subprocess.check_output(
                ['whoami', '/user', '/fo', 'csv', '/nh'], text=True)
            sid = next(csv.reader(identity.strip().splitlines()))[-1]
            if not re.fullmatch(r'S-1-[0-9-]+', sid):
                raise RuntimeError('Could not determine the current Windows identity.')
            subprocess.run(['icacls', str(folder), '/inheritance:r', '/grant:r',
                            f'*{sid}:(OI)(CI)F'], check=True, capture_output=True)
        else:
            folder.chmod(0o700)
        return folder
    except Exception:
        # No credential has been written at this point.
        folder.rmdir()
        raise


def routed_config(config, socks_port, expected_server):
    """Operate only on the selected Ignite cluster; preserve verified TLS."""
    if len(config.get('clusters', [])) != 1 or len(config.get('contexts', [])) != 1:
        raise ValueError('Select one Ignite context with --context.')
    cluster = config['clusters'][0]['cluster']
    url = urlsplit(cluster['server'])
    if (url.scheme != 'https' or url.hostname != API_HOST or
            url.port not in (None, 443) or not re.fullmatch(r'/k8s/clusters/c-[a-z0-9]+/?', url.path) or
            url.username or url.password or url.query or url.fragment):
        raise ValueError('The selected context is not an HTTPS OLCF Rancher cluster endpoint.')
    if cluster['server'].rstrip('/') != expected_server.rstrip('/'):
        raise ValueError('The selected context does not match --expected-server. Verify it in Rancher.')
    if cluster.get('insecure-skip-tls-verify'):
        raise ValueError('TLS verification must be enabled in the kubeconfig.')
    cluster.pop('proxy-url', None)
    if socks_port is not None:
        cluster['proxy-url'] = f'socks5://127.0.0.1:{socks_port}'
    return config


def read_config(args, kubectl):
    command = [kubectl]
    if args.kubeconfig:
        command += ['--kubeconfig', str(Path(args.kubeconfig).expanduser().resolve())]
    if args.context:
        command += ['--context', args.context]
    command += ['config', 'view', '--minify', '--flatten', '--raw', '-o', 'json']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Could not read the selected kubeconfig. Check its path and context.')
    # Never print this output: it contains the user's credentials.
    return json.loads(result.stdout)


def service_ready(port, model):
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/health', timeout=4) as response:
            if response.status != 200:
                return False
        with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=4) as response:
            models = json.load(response)
    except HTTPError as error:
        error.close()
        return False
    return any(item.get('id') == model for item in models.get('data', []))


def test_completion(port, model):
    payload = {'model': model, 'messages': [{'role': 'user',
               'content': 'What is 2 + 2? Answer with only the number.'}],
               'max_tokens': 4096, 'temperature': 0, 'reasoning_effort': 'low'}
    request = Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                      data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=240) as response:
        result = json.load(response)
    answer = result['choices'][0]['message'].get('content')
    if not isinstance(answer, str) or answer.strip() != '4':
        raise RuntimeError('The endpoint responded, but the inference test did not return the expected answer.')
    print('PASS: real chat completion returned 4.', flush=True)


def forward(args, kubectl, config_path, ssh_process):
    environment = dict(os.environ)
    # Do not let a workstation HTTP proxy override the explicit route.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(key, None)
    environment['KUBECTL_PORT_FORWARD_WEBSOCKETS'] = 'true'
    environment['NO_PROXY'] = environment['no_proxy'] = '127.0.0.1,localhost,' + API_HOST
    base = [kubectl, '--kubeconfig', str(config_path), '--request-timeout=20s', '-n', args.namespace]
    delay = 3
    while True:
        free_port(args.port)  # Refuse to reuse someone else's listener.
        events = queue.Queue()
        command = base + ['port-forward', '--address=127.0.0.1', '--pod-running-timeout=30s',
                          'service/' + args.service, f'{args.port}:{args.service_port}']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', env=environment)

        def read_lines():
            for line in process.stdout:
                events.put(line.rstrip())

        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        announced = False
        fatal = None
        began = time.monotonic()
        last_check = 0.0
        try:
            while True:
                while not events.empty():
                    line = events.get_nowait()
                    if not line.startswith('Handling connection for'):
                        print(line, flush=True)
                    if any(term in line.lower() for term in (
                            'unauthorized', 'forbidden', 'provide credentials',
                            'x509:', 'certificate has expired', 'error loading config')):
                        fatal = 'Access failed. Refresh your own Rancher kubeconfig or check namespace permissions.'
                if fatal:
                    raise RuntimeError(fatal)
                if ssh_process is not None and ssh_process.poll() is not None:
                    raise RuntimeError('SSH ended. Run the launcher again and authenticate.')
                if process.poll() is not None:
                    reader.join(timeout=1)
                    # Drain the final error lines before deciding whether to retry.
                    if not events.empty():
                        continue
                    break
                if not announced and time.monotonic() - last_check >= 2:
                    last_check = time.monotonic()
                    try:
                        announced = service_ready(args.port, args.model)
                    except Exception:
                        pass
                    if announced:
                        print(f'\nREADY: http://127.0.0.1:{args.port}/v1\nModel: {args.model}\n'
                              'API key: not required locally (use unused if your client requires one).', flush=True)
                        delay = 3
                        if args.test:
                            test_completion(args.port, args.model)
                        if args.check:
                            print('Check complete; closing this temporary connection.', flush=True)
                            return
                        print('Keep this window open. Ctrl+C disconnects only your connection.\n', flush=True)
                if not announced and time.monotonic() - began > 120:
                    print('Service not ready after two minutes; reconnecting.', flush=True)
                    break
                time.sleep(0.25)
        finally:
            stop(process)
            reader.join(timeout=2)
            if process.stdout:
                process.stdout.close()
        if args.check:
            raise RuntimeError('Connection check failed. The service may be unavailable.')
        print(f'Connection interrupted; retrying in {delay}s. In-flight requests are not replayed.', flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kubeconfig', help='Your own Rancher Ignite kubeconfig; defaults to kubectl configuration')
    parser.add_argument('--context', required=True, help='Explicit Ignite context from your own kubeconfig')
    parser.add_argument('--expected-server', required=True, help='Exact Ignite server URL verified in Rancher')
    parser.add_argument('--namespace', required=True, help='Namespace supplied by the service owner')
    parser.add_argument('--service', required=True, help='Existing Kubernetes Service name')
    parser.add_argument('--service-port', type=int, default=8000)
    parser.add_argument('--model', default='le-chonk', help='Served model ID expected from /v1/models')
    parser.add_argument('--kubectl', default='kubectl', help='kubectl executable or full path (tested: 1.35.6)')
    parser.add_argument('--port', type=int, default=18000, help='Local API port (default: 18000)')
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument('--direct', action='store_true', help='Run where the Rancher API is reachable, normally OLCF home')
    route.add_argument('--ssh-host', help='Your OLCF SSH alias or user@home.ccs.ornl.gov; starts a private SOCKS tunnel')
    route.add_argument('--socks-port', type=int, help='Reuse your existing localhost SSH SOCKS tunnel')
    parser.add_argument('--check', action='store_true', help='Verify the model endpoint, then close only this connection')
    parser.add_argument('--test', action='store_true', help='Also run a small real chat-completion test after connecting')
    args = parser.parse_args()
    for name in (args.namespace, args.service):
        if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', name):
            parser.error('--namespace and --service must be Kubernetes DNS labels')
    if not 1 <= args.service_port <= 65535:
        parser.error('--service-port must be between 1 and 65535')
    if not 1024 <= args.port <= 65535:
        parser.error('--port must be between 1024 and 65535')
    if args.socks_port is not None and not 1024 <= args.socks_port <= 65535:
        parser.error('--socks-port must be between 1024 and 65535')
    kubectl = executable(args.kubectl)
    try:
        free_port(args.port)
    except OSError as error:
        raise RuntimeError(f'Local port {args.port} is already in use or unavailable. '
                           'Keep using the existing connection, or choose another --port.') from error
    config = routed_config(read_config(args, kubectl), args.socks_port, args.expected_server)
    folder = None
    ssh_process = None
    try:
        socks_port = args.socks_port
        if args.ssh_host:
            if args.ssh_host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.ssh_host):
                raise ValueError('Use an SSH alias or user@hostname, without SSH options or whitespace.')
            socks_port = free_port()
            print('Authenticate to OLCF in this terminal when SSH prompts. Your passcode is not saved.', flush=True)
            ssh_process = subprocess.Popen([
                executable('ssh'), '-N', '-T', '-o', 'ExitOnForwardFailure=yes',
                '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
                '-D', f'127.0.0.1:{socks_port}', args.ssh_host])
            deadline = time.monotonic() + 300
            while not check_socks(socks_port):
                if ssh_process.poll() is not None:
                    raise RuntimeError('SSH authentication or connection failed.')
                if time.monotonic() > deadline:
                    raise RuntimeError('Timed out waiting for SSH authentication.')
                time.sleep(0.5)
        elif socks_port is not None and not check_socks(socks_port):
            raise RuntimeError(f'No working SOCKS5 tunnel on 127.0.0.1:{socks_port}.')
        config = routed_config(config, socks_port, args.expected_server)
        folder = private_directory()
        config_path = folder / 'ignite.json'
        with config_path.open('x', encoding='utf-8') as handle:
            json.dump(config, handle)
        if os.name != 'nt':
            config_path.chmod(0o600)
        forward(args, kubectl, config_path, ssh_process)
    finally:
        stop(ssh_process)
        if folder is not None:
            # Only remove the temporary directory created by this invocation.
            assert folder.parent == Path(tempfile.gettempdir()).resolve()
            assert folder.name.startswith('le-chonk-')
            shutil.rmtree(folder)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nDisconnected. The shared model remains running.')
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
