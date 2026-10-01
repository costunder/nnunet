"""CUDA UNIT audit of the current prompt operator, using embedding fixtures only.

No CT, CNN validation, optimizer update, production checkpoint, or ready marker.
The joint reference uses the CURRENT local RelationLayer equations; this is
not an execution or reproduction of the official PRODIGY operator.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v222.contracts import load_config
from hiercp_v222.model import PromptGraphModel
from hiercp_v222.v1_cache import configuration


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(net):
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        digest.update(name.encode())
        if not torch.is_tensor(value):
            raise TypeError('This UNIT fixture must have tensor-only model state')
        tensor = value.detach().contiguous().cpu()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def modes(net):
    return [(name, module.training) for name, module in net.named_modules()]


def l1_parameters(net):
    return [(name, value) for name, value in net.named_parameters()
            if name == 'label_seed' or name.startswith('l1.')]


def support_edges(owners, classes):
    n = len(owners)
    data = torch.arange(n, device=owners.device).repeat_interleave(2)
    cls = torch.arange(2, device=owners.device).repeat(n)
    labels = n + owners.repeat_interleave(2)*2 + cls
    known = torch.stack((torch.ones_like(cls), (classes.repeat_interleave(2) == cls).long()), -1).float()
    return torch.stack((torch.cat((data, labels)), torch.cat((labels, data)))), torch.cat((known, known))


def split_l1(net, support, query, owners, classes, physical=32):
    histories, final_labels = net.encode_support(support, owners, classes)
    outputs = []
    for value in query.split(physical):
        q = value
        for layer, labels in zip(net.l1, histories):
            src = torch.arange(len(labels), device=q.device).repeat(len(q))
            dst = torch.arange(len(q), device=q.device).repeat_interleave(len(labels))
            q = layer.messages(labels, q, src, dst, q.new_zeros(len(src), 2))
        outputs.append(q)
    return torch.cat(outputs), final_labels, histories


def joint_l1(net, support, query, owners, classes):
    """Joint current-operator graph: identical support edges and query sinks."""
    n, p = len(support), int(owners.max())+1
    edges, attributes = support_edges(owners, classes)
    label_ids = torch.arange(n, n+p*2, device=query.device)
    src = label_ids.repeat(len(query))
    dst = torch.arange(n+p*2, n+p*2+len(query), device=query.device).repeat_interleave(p*2)
    edges = torch.cat((edges, torch.stack((src, dst))), 1)
    attributes = torch.cat((attributes, query.new_zeros(len(src), 2)))
    x = torch.cat((support, net.label_seed.repeat(p, 1), query))
    histories = []
    for layer in net.l1:
        histories.append(x[n:n+p*2])
        x = layer(x, edges, attributes)
    return x[n+p*2:], x[n:n+p*2].reshape(p, 2, net.dim), histories


def close(left, right, *, atol=2e-6, rtol=2e-5):
    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    return float((left-right).abs().max())


def derivative_parity(net, support, query, owners, classes):
    coefficients_q = torch.randn_like(query)
    coefficients_l = torch.randn((16, 2, 128), device=query.device)
    params = l1_parameters(net)
    results = []
    for function in (split_l1, joint_l1):
        s = support.detach().clone().requires_grad_()
        q = query.detach().clone().requires_grad_()
        y, labels, _ = function(net, s, q, owners, classes)
        objective = (y*coefficients_q).mean()+(labels*coefficients_l).mean()
        derivatives = torch.autograd.grad(objective, [s, q]+[value for _, value in params])
        if any(not bool(torch.isfinite(value).all()) for value in derivatives):
            raise FloatingPointError('Nonfinite split/joint derivative')
        results.append(derivatives)
    names = ['support_input', 'query_input']+[name for name, _ in params]
    deltas = {name: close(a, b, atol=3e-6, rtol=4e-4)
              for name, a, b in zip(names, *results)}
    return dict(all_compared_parameters=len(params), derivative_count=len(names),
                max_abs_differences=deltas, atol=3e-6, rtol=4e-4,
                objective='same random linear L1 query and final-label readout',
                mode='eval; all stochastic dropout disabled')


def capture_topology(net, support, query, owners, classes):
    captured = []
    original = net.l1[0].forward

    def forward(nodes, edges, features):
        captured.append((edges.detach().clone(), features.detach().clone()))
        return original(nodes, edges, features)

    with torch.no_grad(), patch.object(net.l1[0], 'forward', side_effect=forward):
        net.encode_support(support, owners, classes)
    if len(captured) != 1:
        raise AssertionError('Support topology was not captured from actual encode_support')
    edges, features = captured[0]
    expected_e, expected_f = support_edges(owners, classes)
    if not torch.equal(edges, expected_e) or not torch.equal(features, expected_f):
        raise AssertionError('Actual support T/F patient binding differs')
    if bool((edges[0] == edges[1]).any()):
        raise AssertionError('Current support path unexpectedly adds self-loops')
    with torch.no_grad():
        state = net.prepare_support(support, owners, classes)
    observed = []
    patches = []
    for index, layer in enumerate(net.l1):
        method = layer.messages

        def messages(source, destination, src, dst, attrs, method=method, index=index):
            observed.append(dict(index=index, source=source, destination=destination,
                                 src=src.detach().clone(), dst=dst.detach().clone(),
                                 attrs=attrs.detach().clone()))
            return method(source, destination, src, dst, attrs)
        patches.append(patch.object(layer, 'messages', side_effect=messages))
    from contextlib import ExitStack
    with torch.no_grad(), ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        output = net.predict_embeddings(query[:32], state)
    if len(observed) != 2 or output['logits'].shape != (32, 2):
        raise AssertionError('Actual predict path does not execute exactly two L1 layers')
    for item in observed:
        index = item['index']
        if item['source'] is not state['histories'][index]:
            raise AssertionError('Query source is not the cached layer-input labels')
        if len(item['source']) != 32 or len(item['destination']) != 32:
            raise AssertionError('Actual query source/destination population differs')
        src = torch.arange(32, device=query.device).repeat(32)
        dst = torch.arange(32, device=query.device).repeat_interleave(32)
        if (not torch.equal(item['src'], src) or not torch.equal(item['dst'], dst)
                or bool(item['attrs'].ne(0).any())):
            raise AssertionError('Query must contain label-to-query U incoming edges only')
    return dict(support_nodes=256, patient_labels=32, support_edges=int(edges.shape[1]),
                support_edge_codes={'T': [1, 1], 'F': [1, 0]},
                query_edge_code=[0, 0], query_incoming_edges_per_candidate=32,
                query_edges_per_physical_chunk=1024,
                support_self_loops=False, query_reverse_edges=False,
                query_self_loops=False, captured_from_actual_forward_and_predict=True,
                note='source and destination indices use separate node spaces; numerical src==dst is not a self-loop')


def run(report):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this explicit CUDA UNIT; no CPU fallback')
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    cfg, base = configuration()
    default_cfg, default_base = load_config()
    for key in ('task_layers', 'alignment_layers', 'label_count', 'temperature', 'alignment_loss_weight'):
        if cfg[key] != default_cfg[key]:
            raise AssertionError(f'Loaded approved prompt configurations disagree on {key}')
    if base['model'] != default_base['model']:
        raise AssertionError('Loaded base model configurations disagree')
    if (base['model']['hidden_dim'], base['model']['heads'], cfg['task_layers'], cfg['alignment_layers']) != (128, 4, 2, 2):
        raise AssertionError('Full approved L1/L2 scale required')
    net = PromptGraphModel(cfg, base, {}, local_encoder=nn.Identity()).cuda().eval()
    before, before_modes = state_hash(net), modes(net)
    before_grad = [parameter.grad for parameter in net.parameters()]
    sources = ['hiercp_v222/model.py', 'hiercp_v222/clustering.py',
               'hiercp_v222/contracts.py', 'hiercp_v222/v1_cache.py',
               'config/prompt_graph_v222.json', 'config/prompt_graph_v222_v1_l0.json',
               'config/train.json']
    hashes = {name: file_hash(ROOT/name) for name in sources}
    support = torch.randn(256, 128, device='cuda')
    query = torch.randn(128, 128, device='cuda')
    owners = torch.arange(16, device='cuda').repeat_interleave(16)
    classes = torch.arange(16, device='cuda').remainder(2).repeat(16)
    report['resources_before'] = dict(gpu=torch.cuda.get_device_name(), device_total_bytes=total,
        device_free_bytes=free, cpu_logical=psutil.cpu_count(), ram_available=psutil.virtual_memory().available,
        process_rss=psutil.Process().memory_info().rss,
        nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                                           '--format=csv,noheader'], text=True).strip())
    report['model'] = dict(hidden_dim=128, heads=4, L1_layers=2, L2_layers=2,
        label_count=2, prompt_parameters=sum(p.numel() for p in net.parameters()),
        local_encoder='nn.Identity injected solely for embedding-fixture L1/L2 UNIT',
        precision='FP32', physical_query_chunk=32, query_candidates=128,
        support_patients=16, support_observations_per_patient=16, support_observations=256,
        optimizer=None, gradient_accumulation=None, effective_training_batch=None,
        CNN_tested=False, CT_used=False)
    report['checks']['actual_topology'] = capture_topology(net, support, query, owners, classes)
    with torch.no_grad():
        split_q, split_l, split_h = split_l1(net, support, query, owners, classes)
        joint_q, joint_l, joint_h = joint_l1(net, support, query, owners, classes)
        report['checks']['synchronous_two_layer_parity'] = dict(
            query_max_abs=close(split_q, joint_q), label_max_abs=close(split_l, joint_l),
            history_max_abs=[close(a, b) for a, b in zip(split_h, joint_h)],
            history0_matches_shared_seeds=bool(torch.equal(split_h[0], net.label_seed.repeat(16, 1))))
    report['checks']['input_and_parameter_gradient_parity'] = derivative_parity(net, support, query, owners, classes)
    with torch.no_grad():
        state = net.prepare_support(support, owners, classes)
        chunked = torch.cat([net.predict_embeddings(value, state)['logits'] for value in query.split(32)])
        whole = net.predict_embeddings(query, state)['logits']
        order = torch.randperm(128, device='cuda')
        permuted = torch.cat([net.predict_embeddings(value, state)['logits'] for value in query[order].split(32)])
        report['checks']['eval_chunk_permutation_parity'] = dict(
            chunk_vs_whole_max_abs=close(chunked, whole),
            permutation_max_abs=close(permuted[order.argsort()], chunked),
            all_128_candidates_retained=True, actual_chunk_sizes=[32, 32, 32, 32])
    q = query.detach().clone().requires_grad_()
    support_leaf = support.detach().clone().requires_grad_()
    state = net.prepare_support(support_leaf.detach(), owners, classes)
    logits = torch.cat([net.predict_embeddings(value, state)['logits'] for value in q.split(32)])
    score = logits[:, 1]-logits[:, 0]
    truth = torch.arange(128, device='cuda').remainder(2).bool()
    rank_loss = F.softplus(score[~truth, None]-score[None, truth]).mean()
    names, parameters = zip(*net.named_parameters())
    derivatives = torch.autograd.grad(rank_loss, [q, support_leaf]+list(parameters), allow_unused=True)
    if derivatives[0] is None or not bool(torch.isfinite(derivatives[0]).all()) or not bool(derivatives[0].ne(0).any()):
        raise AssertionError('Live query rank gradient missing, zero, or nonfinite')
    if derivatives[1] is not None:
        raise AssertionError('Epoch detached support unexpectedly receives input gradient')
    norms = {}
    for label, prefixes in [('L1', ('l1.',)), ('L2', ('l2.', 'l2_updates.')), ('label_seed', ('label_seed',))]:
        values = [value for name, value in zip(names, derivatives[2:]) if name.startswith(prefixes)]
        if not values or any(value is None or not bool(torch.isfinite(value).all()) for value in values):
            raise AssertionError(f'{label} rank derivative is missing or nonfinite')
        norm = float(torch.stack([value.square().sum() for value in values]).sum().sqrt())
        if norm <= 0:
            raise AssertionError(f'{label} rank derivative is zero')
        norms[label] = norm
    report['checks']['detached_support_live_query_rank_gradient'] = dict(
        mode='eval stochastic dropout off; same production detach/data-flow contract',
        query_norm=float(derivatives[0].norm()), support_gradient=None,
        module_gradient_norms=norms, rank_loss=float(rank_loss.detach()),
        query_candidates=128, pair_comparisons=4096, optimizer_updates=0,
        meaning='gradient connectivity only; random UNIT class fixture is not learning performance')
    refused = False
    try:
        net.predict_embeddings(q, state, query_targets=truth.long())
    except TypeError as error:
        if 'query_targets' not in str(error):
            raise
        refused = True
    if not refused or 'query_targets' in inspect.signature(net.forward).parameters:
        raise AssertionError('Query targets may not enter prediction/forward API')
    report['checks']['query_targets_rejected_by_forward_api'] = True
    unchanged = state_hash(net) == before and modes(net) == before_modes
    if not unchanged or any(p.grad is not old for p, old in zip(net.parameters(), before_grad)):
        raise AssertionError('UNIT audit mutated model state/modes/gradient buffers')
    if hashes != {name: file_hash(ROOT/name) for name in sources}:
        raise AssertionError('Production source changed during audit')
    report['checks']['state_and_source_preserved'] = dict(
        model_state_sha256=before, state_preserved=True, modes_preserved=True,
        gradient_buffers_preserved=True, source_hashes=hashes)
    torch.cuda.synchronize()
    report['resources_after'] = dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), process_rss=psutil.Process().memory_info().rss,
        ram_available=psutil.virtual_memory().available)
    report['status'] = 'PASS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Existing UNIT evidence cannot be overwritten: '+str(output))
    report = dict(scope='CUDA UNIT, synthetic embedding fixtures; no CT/CNN/trained performance',
        diagnostic_only=True, official_PRODIGY_execution=False,
        reference='joint graph executes the same CURRENT local RelationLayer equations',
        production_optimizer_updates=0, production_checkpoint_written=False, production_ready=False,
        script_sha256=file_hash(__file__), checks={},
        software=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, cuda_build=torch.version.cuda))
    began = time.perf_counter()
    error = None
    try:
        run(report)
    except Exception as caught:
        error = caught
        report['status'] = 'FAIL'
        report['error'] = dict(type=type(caught).__name__, message=str(caught))
    finally:
        report['elapsed_seconds'] = time.perf_counter()-began
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(dict(status=report['status'], scope=report['scope'], checks=list(report['checks']),
                              seconds=report['elapsed_seconds'], report=str(output)), ensure_ascii=False), flush=True)
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
