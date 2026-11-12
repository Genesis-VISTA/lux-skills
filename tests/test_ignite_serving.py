"""Offline checks for the Ignite client and deployment-plan isolation."""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/ignite-serving-skill/scripts'


def load(name):
    spec = importlib.util.spec_from_file_location('ignite_' + name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load('prepare_ml4')
client = load('connect')


def fixture():
    labels = {'app.kubernetes.io/name': 'zenith-ml4', 'app.kubernetes.io/component': 'server'}
    return {'apiVersion': 'v1', 'kind': 'List', 'items': [
        {'kind': 'ConfigMap', 'metadata': {'name': 'ml4-script'}, 'data': {'script': 'example'}},
        {'kind': 'Service', 'metadata': {'name': 'ml4'}, 'spec': {'selector': dict(labels)}},
        {'kind': 'Job', 'metadata': {'name': 'trial'}, 'spec': {'template': {
            'metadata': {'labels': dict(labels)}, 'spec': {
                'volumes': [{'name': 'script', 'configMap': {'name': 'ml4-script'}},
                            {'name': 'models', 'persistentVolumeClaim': {'claimName': 'trial-models'}}]
            }}}}
    ]}


def selects(selector, labels):
    return all(labels.get(key) == value for key, value in selector.items())


class PlanTests(unittest.TestCase):
    def test_service_isolation_including_original_broad_selector(self):
        original = fixture()
        before = copy.deepcopy(original)
        first, second = plan.isolate(original, 'trial-a'), plan.isolate(original, 'trial-b')
        service = first['items'][1]
        labels = first['items'][2]['spec']['template']['metadata']['labels']
        other_labels = second['items'][2]['spec']['template']['metadata']['labels']
        self.assertTrue(selects(service['spec']['selector'], labels))
        self.assertFalse(selects(service['spec']['selector'], other_labels))
        self.assertFalse(selects(original['items'][1]['spec']['selector'], labels))
        self.assertEqual(original, before)

    def test_configmap_references_follow_names_and_pvcs_are_unchanged(self):
        result = plan.isolate(fixture(), 'trial-a')['items']
        self.assertEqual(result[0]['metadata']['name'], 'trial-a-ml4-script')
        volumes = result[2]['spec']['template']['spec']['volumes']
        self.assertEqual(volumes[0]['configMap']['name'], result[0]['metadata']['name'])
        self.assertEqual(volumes[1]['persistentVolumeClaim']['claimName'], 'trial-models')

    def test_reject_unreviewed_or_dirty_checkout(self):
        with patch.object(plan.subprocess, 'check_output', return_value='wrong-commit\n'):
            with self.assertRaisesRegex(ValueError, 'reviewed Zenith commit'):
                plan.checked_checkout('.')
        with patch.object(plan.subprocess, 'check_output', side_effect=[plan.ZENITH_COMMIT, ' M changed.py']):
            with self.assertRaisesRegex(ValueError, 'clean Zenith checkout'):
                plan.checked_checkout('.')

    def test_plan_hashes_and_existing_directory_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'plan'
            args = SimpleNamespace(namespace='allocated-gpu', name='trial', serve_hours=4)
            summary = plan.write_plan(target, {'serve': fixture()}, args)
            saved = (target / 'serve.json').read_bytes()
            self.assertEqual(summary['sha256']['serve.json'], hashlib.sha256(saved).hexdigest())
            self.assertEqual(json.loads((target / 'plan.json').read_text())['max_serving_gpu_hours'], 16)
            with self.assertRaises(FileExistsError):
                plan.write_plan(target, {'serve': {}}, args)
            self.assertEqual((target / 'serve.json').read_bytes(), saved)

    def test_bounds_fail_before_checkout_or_renderer(self):
        base = dict(name='trial', namespace='allocated-gpu', serve_hours=4)
        for field, value in [('name', 'A'), ('name', 'x' * 31), ('namespace', 'bad/name'),
                             ('serve_hours', 0), ('serve_hours', 49)]:
            with self.subTest(field=field, value=value):
                args = SimpleNamespace(**dict(base, **{field: value}))
                with patch.object(plan, 'checked_checkout') as checkout:
                    with self.assertRaises(ValueError):
                        plan.render_plan(args)
                    checkout.assert_not_called()


class ClientConfigTests(unittest.TestCase):
    server = 'https://console.apps.slate-mod.ccs.ornl.gov:443/k8s/clusters/c-test1'

    def config(self):
        return {'clusters': [{'name': 'ignite', 'cluster': {
            'server': self.server, 'certificate-authority-data': 'TEST-CA',
            'proxy-url': 'http://old.invalid:3128'}}],
            'contexts': [{'name': 'ignite', 'context': {'cluster': 'ignite', 'user': 'self'}}],
            'users': [{'name': 'self', 'user': {'token': 'FAKE-UNIT-TEST-TOKEN'}}]}

    def test_routing_retains_tls_and_identity(self):
        original = self.config()
        result = client.routed_config(copy.deepcopy(original), 10000, self.server)
        cluster = result['clusters'][0]['cluster']
        self.assertEqual(cluster['proxy-url'], 'socks5://127.0.0.1:10000')
        self.assertEqual(cluster['certificate-authority-data'], 'TEST-CA')
        self.assertEqual(result['users'], original['users'])
        self.assertEqual(cluster['server'], self.server)
        self.assertNotIn('proxy-url', client.routed_config(result, None, self.server)['clusters'][0]['cluster'])

    def test_reject_wrong_cluster_or_disabled_tls(self):
        for server in ['http://console.apps.slate-mod.ccs.ornl.gov/k8s/clusters/c-test1',
                       'https://attacker.invalid/k8s/clusters/c-test1',
                       self.server + '?token=example', self.server.replace(':443', ':444'),
                       self.server.replace('c-test1', 'c-other')]:
            with self.subTest(server=server):
                config = self.config()
                config['clusters'][0]['cluster']['server'] = server
                with self.assertRaises(ValueError):
                    client.routed_config(config, 10000, self.server)
        config = self.config()
        config['clusters'][0]['cluster']['insecure-skip-tls-verify'] = True
        with self.assertRaises(ValueError):
            client.routed_config(config, None, self.server)
        config = self.config()
        config['contexts'] *= 2
        with self.assertRaises(ValueError):
            client.routed_config(config, None, self.server)

    def test_occupied_port_is_not_reused(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            with self.assertRaises(OSError):
                client.free_port(listener.getsockname()[1])


class ProbeTests(unittest.TestCase):
    def setUp(self):
        owner = self
        self.answer = '4'
        self.health = 200
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append(self.path)
                self.send_response(owner.health if self.path == '/health' else 200)
                self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': 'le-chonk'}]}).encode())

            def do_POST(self):
                owner.requests.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'message': {'content': owner.answer}}]}).encode())

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_health_and_exact_model_are_required(self):
        self.assertTrue(client.service_ready(self.port, 'le-chonk'))
        self.assertEqual(self.requests[:2], ['/health', '/v1/models'])
        self.assertFalse(client.service_ready(self.port, 'other-model'))
        self.health = 503
        self.assertFalse(client.service_ready(self.port, 'le-chonk'))

    def test_real_http_completion_and_rejection_of_empty_or_wrong_answers(self):
        client.test_completion(self.port, 'le-chonk')
        self.assertEqual(self.requests[-1][0], '/v1/chat/completions')
        self.assertEqual(self.requests[-1][1]['model'], 'le-chonk')
        for answer in ['', None, '5']:
            with self.subTest(answer=answer):
                self.answer = answer
                with self.assertRaisesRegex(RuntimeError, 'expected answer'):
                    client.test_completion(self.port, 'le-chonk')


if __name__ == '__main__':
    unittest.main()
