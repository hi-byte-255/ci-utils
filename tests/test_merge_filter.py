"""Regression tests for the wave merge/filter stage.

The wave-3 filter kept hosts with >=10 open ports (firewalls) and discarded the
real servers. These tests pin the corrected direction so it cannot regress.
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import merge_filter as mf


def test_middlebox_with_10_ports_is_dropped():
    ip_ports = {"1.1.1.1": set(range(1, 11))}  # 10 ports open -> firewall
    candidates, middleboxes = mf.split_candidates(ip_ports)
    assert "1.1.1.1" not in candidates, "10-port host must NOT be a candidate"
    assert "1.1.1.1" in middleboxes


def test_gateway_with_few_ports_is_kept():
    ip_ports = {"2.2.2.2": {8080, 3000, 8000}}  # wave-3 gateway shape
    candidates, middleboxes = mf.split_candidates(ip_ports)
    assert "2.2.2.2" in candidates
    assert "2.2.2.2" not in middleboxes


def test_threshold_boundary_is_exclusive_of_candidates():
    ip_ports = {
        "3.3.3.3": set(range(9)),           # 9 ports -> candidate (<10)
        "4.4.4.4": set(range(10)),          # 10 ports -> middlebox
    }
    candidates, middleboxes = mf.split_candidates(ip_ports)
    assert set(candidates) == {"3.3.3.3"}
    assert set(middleboxes) == {"4.4.4.4"}


def test_inverted_wave3_predicate_would_fail_this_assertion():
    """Guard: the old `len(ps) >= MIN_PORTS` kept firewalls, not servers."""
    ip_ports = {"5.5.5.5": set(range(12)), "6.6.6.6": {8080}}
    # corrected behaviour
    candidates, _ = mf.split_candidates(ip_ports)
    assert candidates == {"6.6.6.6": {8080}}
    # the old (buggy) direction would have produced the opposite
    old_kept = {ip for ip, ps in ip_ports.items() if len(ps) >= 10}
    assert old_kept == {"5.5.5.5"}
    assert old_kept != set(candidates)


def test_pick_probe_port_prefers_gateway_port():
    assert mf.pick_probe_port({9200, 8080, 5000}) == 8080
    assert mf.pick_probe_port({3000, 8000}) == 3000
    assert mf.pick_probe_port({2375}) == 2375


def test_load_hits_tolerates_trailing_comma(tmp_path):
    p = tmp_path / "hits_0.json"
    p.write_text('[{"ip": "7.7.7.7", "ports": [{"port": 8080, "status": "open"}]},\n')
    recs = mf.load_hits(str(p))
    assert recs and recs[0]["ip"] == "7.7.7.7"


def test_end_to_end_keeps_only_low_port_hosts(tmp_path):
    hits = tmp_path / "hits"
    (hits / "shard_0").mkdir(parents=True)
    (hits / "shard_0" / "hits_0.json").write_text(json.dumps([
        {"ip": "8.8.8.8", "ports": [{"port": p, "status": "open"} for p in range(1, 16)]},
        {"ip": "9.9.9.9", "ports": [{"port": 8080, "status": "open"},
                                    {"port": 3000, "status": "open"}]},
    ]))
    out = tmp_path / "out"
    os.environ["HITS_DIR"] = str(hits)
    os.environ["OUT_DIR"] = str(out)
    importlib.reload(mf)
    assert mf.main() == 0

    pass1 = (out / "hosts_pass1.txt").read_text().splitlines()
    assert pass1 == ["9.9.9.9:8080", "9.9.9.9:3000"], pass1  # firewall excluded, all high-value ports emitted in priority order

    clean = [json.loads(l) for l in (out / "candidates_clean.jsonl").read_text().splitlines()]
    assert [c["ip"] for c in clean] == ["9.9.9.9"]

    mid = [json.loads(l) for l in (out / "middleboxes.jsonl").read_text().splitlines()]
    assert [m["ip"] for m in mid] == ["8.8.8.8"]


def test_proxy_ip_and_connect_guards():
    import asyncio
    import validate_v2 as v2
    assert v2._is_valid_public_ip("8.8.8.8") is True
    assert v2._is_valid_public_ip("127.0.0.1") is False
    assert v2._is_valid_public_ip("10.0.0.1") is False
    assert v2._is_valid_public_ip("0.0.0.0") is False
    # HTTP proxy without CONNECT support must be rejected
    res = asyncio.run(v2._classify_exit_ip(None, "1.2.3.4", "8.8.8.8", proto="http", https_connect=False))
    assert res is None
    # Private exit IP must be rejected even on SOCKS5
    res2 = asyncio.run(v2._classify_exit_ip(None, "1.2.3.4", "127.0.0.1", proto="socks5", https_connect=True))
    assert res2 is None


def test_sweep4_ports_and_multi_emit_alignment():
    """Every port scanned in sweep4.yml must be in PORT_PRIORITY and MULTI_EMIT_PORTS,
    and sweep4.yml must not scan 11434 when validate_v2 drops it by default."""
    wf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".github", "workflows", "sweep4.yml")
    sweep_ports = []
    with open(wf_path) as f:
        for line in f:
            if "default:" in line and "8000,8080" in line:
                sweep_ports = [int(x) for x in line.split("'")[1].split(",")]
                break
    assert sweep_ports, "Failed to parse sweep4.yml default ports"
    assert 11434 not in sweep_ports, "11434 is dropped by validate_v2 default Zero-Ollama guard"
    assert 23333 in sweep_ports, "23333 (LMDeploy) should be in sweep4.yml"
    assert 5678 in mf.MULTI_EMIT_PORTS, "5678 must be emitted by pick_probe_ports"
    missing_prio = [p for p in sweep_ports if p not in mf.PORT_PRIORITY]
    missing_emit = [p for p in sweep_ports if p not in mf.MULTI_EMIT_PORTS]
    assert not missing_prio, f"Missing from PORT_PRIORITY: {missing_prio}"
    assert not missing_emit, f"Missing from MULTI_EMIT_PORTS: {missing_emit}"


def test_control_plane_https_and_sidecar_contracts():
    """Verify HTTPS scheme on 6443/10250, etcd range_end query, k8s anon_api gate, and Docker registry :5000."""
    import asyncio
    import validate_v2 as v2

    class MockResp:
        def __init__(self, status=200, json_data=None, text_data="OK", headers=None):
            self.status = status
            self._json = json_data
            self._text = text_data
            self.headers = headers or {}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def json(self, content_type=None):
            return self._json
        async def text(self):
            return self._text

    class RouteSession:
        def __init__(self, routes):
            self.routes = routes
            self.calls = []
        def get(self, url, **kw):
            self.calls.append(("GET", url, kw))
            return self.routes.get(("GET", url), MockResp(status=404))
        def post(self, url, **kw):
            self.calls.append(("POST", url, kw))
            return self.routes.get(("POST", url), MockResp(status=404))

    # 1. kubelet 10250 must probe https:// by default in validate_host
    s_kube = RouteSession({
        ("GET", "https://1.2.3.4:10250/pods"): MockResp(200, {"kind": "PodList", "items": [{"metadata": {"name": "p1"}}]}),
    })
    res_kube = asyncio.run(v2.validate_host(s_kube, "1.2.3.4:10250", 5.0, 0, asyncio.Semaphore(4)))
    assert res_kube["tier"] == "KUBELET_API", res_kube
    assert res_kube["sidecar"]["pods"] == 1

    # 2. k8s 6443 with 403 on /api/v1/namespaces must NOT report as open K8S_API_SERVER
    s_k8s_locked = RouteSession({
        ("GET", "https://1.2.3.4:6443/version"): MockResp(200, {"gitVersion": "v1.30.0"}),
        ("GET", "https://1.2.3.4:6443/api/v1/namespaces"): MockResp(403, {"status": "Failure"}),
        ("GET", "https://1.2.3.4:6443/api/v1/pods?limit=1"): MockResp(403, {"status": "Failure"}),
    })
    res_k8s_locked = asyncio.run(v2.validate_host(s_k8s_locked, "1.2.3.4:6443", 5.0, 0, asyncio.Semaphore(4)))
    assert res_k8s_locked["tier"] == "DEAD", f"Locked 6443 must not false-positive: {res_k8s_locked}"

    # 3. k8s 6443 with 200 on /api/v1/namespaces (anon RBAC open) MUST report K8S_API_SERVER
    s_k8s_open = RouteSession({
        ("GET", "https://1.2.3.4:6443/version"): MockResp(200, {"gitVersion": "v1.30.0"}),
        ("GET", "https://1.2.3.4:6443/api/v1/namespaces"): MockResp(200, {"kind": "NamespaceList", "items": [{"metadata": {"name": "default"}}]}),
    })
    res_k8s_open = asyncio.run(v2.validate_host(s_k8s_open, "1.2.3.4:6443", 5.0, 0, asyncio.Semaphore(4)))
    assert res_k8s_open["tier"] == "K8S_API_SERVER", res_k8s_open
    assert res_k8s_open["sidecar"]["anon_api"] is True

    # 4. etcd 2379 must pass range_end="AA==" so all keys are counted, not just literal \x00
    s_etcd = RouteSession({
        ("POST", "http://1.2.3.4:2379/v3/kv/range"): MockResp(200, {"header": {}, "count": "42"}),
        ("GET", "http://1.2.3.4:2379/version"): MockResp(200, text_data='{"etcdserver":"3.5.12"}'),
    })
    res_etcd = asyncio.run(v2._sidecar_triage(s_etcd, "1.2.3.4", 2379, "http://1.2.3.4:2379"))
    assert res_etcd and res_etcd["sidecar"] == "OPEN_ETCD", res_etcd
    post_call = [c for c in s_etcd.calls if c[0] == "POST"][0]
    assert post_call[2]["json"].get("range_end") == "AA==", f"etcd range_end missing: {post_call[2]}"
    assert res_etcd.get("keys") == 42

    # 5. Docker Registry v2 on :5000 when LLM endpoints fail
    s_reg = RouteSession({
        ("GET", "http://1.2.3.4:5000/v2/_catalog"): MockResp(200, {"repositories": ["app/backend", "app/worker"]}),
    })
    res_reg = asyncio.run(v2._sidecar_triage(s_reg, "1.2.3.4", 5000, "http://1.2.3.4:5000"))
    assert res_reg and res_reg["sidecar"] == "OPEN_DOCKER_REGISTRY", res_reg
    assert res_reg["repos"] == 2


