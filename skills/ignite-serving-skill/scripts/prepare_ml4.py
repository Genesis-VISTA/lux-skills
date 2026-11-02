#!/usr/bin/env python3
"""Render an isolated ML4 plan from a pinned, authorized Zenith checkout; no API writes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ZENITH_COMMIT = 'ebf19b355b753d5a26c590463f782a0f1f43901f'
PHASES = ('storage', 'build', 'download', 'preflight', 'serve')


def checked_checkout(path):
    path = Path(path).expanduser().resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != ZENITH_COMMIT:
        raise ValueError('Use the reviewed Zenith commit ' + ZENITH_COMMIT)
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Use a clean Zenith checkout so rendered source provenance is meaningful')
    renderer = path / 'deploy/ml4/pvc_render.py'
    if not renderer.is_file():
        raise ValueError('This checkout lacks the PVC renderer from Zenith PR #2')
    return path, renderer


def isolate(manifest, name):
    """Keep one named service from selecting another deployment's GPU pods."""
    manifest = copy.deepcopy(manifest)
    items = manifest['items']
    maps = {item['metadata']['name']: name + '-' + item['metadata']['name']
            for item in items if item['kind'] == 'ConfigMap'}
    instance = {'app.kubernetes.io/name': 'ignite-ml4-' + name,
                'app.kubernetes.io/instance': name}
    for item in items:
        metadata = item['metadata']
        metadata.setdefault('labels', {}).update(instance)
        if item['kind'] == 'ConfigMap':
            metadata['name'] = maps[metadata['name']]
        if item['kind'] == 'Service':
            metadata['name'] = name
            item['spec']['selector'].update(instance)
        if item['kind'] == 'Job':
            template = item['spec']['template']
            template.setdefault('metadata', {}).setdefault('labels', {}).update(instance)
            for volume in template['spec'].get('volumes', []):
                if 'configMap' in volume:
                    volume['configMap']['name'] = maps[volume['configMap']['name']]
    return manifest


def render_plan(args):
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,28}[a-z0-9])?', args.name):
        raise ValueError('--name must be a DNS label of at most 30 characters')
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', args.namespace):
        raise ValueError('--namespace must be your assigned GPU namespace')
    if not 1 <= args.serve_hours <= 48:
        raise ValueError('--serve-hours must be between 1 and 48')
    _, renderer = checked_checkout(args.zenith)
    common = ['--namespace', args.namespace, '--build-pvc', args.name + '-build',
              '--pvc', args.name + '-models', '--storage-class', args.storage_class]
    for option, value in [('--proxy', args.proxy), ('--pull-secret', args.pull_secret),
                          ('--hf-secret', args.hf_secret)]:
        if value:
            common += [option, value]
    result = {}
    for phase in PHASES:
        command = [sys.executable, '-B', str(renderer), phase, *common]
        if phase != 'storage':
            command += ['--job-name', args.name if phase == 'serve' else args.name + '-' + phase]
        if phase == 'serve':
            command += ['--walltime-seconds', str(args.serve_hours * 3600)]
        payload = subprocess.check_output(command, text=True)
        result[phase] = isolate(json.loads(payload), args.name)
    return result


def write_plan(output, manifests, args):
    # All rendering succeeds before any output is written. Never overwrite a plan.
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for phase, manifest in manifests.items():
        content = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
        (output / (phase + '.json')).write_bytes(content)
        checksums[phase + '.json'] = hashlib.sha256(content).hexdigest()
    summary = {'zenith_commit': ZENITH_COMMIT, 'namespace': args.namespace, 'name': args.name,
               'service': args.name, 'model': 'le-chonk', 'serve_hours': args.serve_hours,
               'serving_gpus': 4, 'max_serving_gpu_hours': 4 * args.serve_hours,
               'build_pvc': args.name + '-build', 'model_pvc': args.name + '-models',
               'sha256': checksums}
    (output / 'plan.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zenith', required=True, type=Path, help='Clean checkout of the pinned Zenith contribution')
    parser.add_argument('--namespace', required=True, help='Actual allocated GPU namespace, not a guessed project suffix')
    parser.add_argument('--name', required=True, help='Unique workload name, at most 30 characters')
    parser.add_argument('--out', required=True, type=Path, help='New output directory; keep outside version control')
    parser.add_argument('--serve-hours', type=int, default=4, help='Bounded test session, 1-48 hours (default 4)')
    parser.add_argument('--storage-class', default='netapp-file')
    parser.add_argument('--proxy', help='OLCF egress proxy for build/staging, without embedded credentials')
    parser.add_argument('--pull-secret', help='Existing image-pull Secret name')
    parser.add_argument('--hf-secret', help='Optional existing HF Secret name, with token key; anonymous by default')
    args = parser.parse_args()
    if args.out.expanduser().exists():
        parser.error('--out must be a new directory; existing plans are never overwritten')
    manifests = render_plan(args)
    print(json.dumps(write_plan(args.out, manifests, args), indent=2))


if __name__ == '__main__':
    main()
