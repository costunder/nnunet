"""Explicit DEBUG fixtures: execution equivalence, not CT efficacy or epoch speed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from contextlib import nullcontext
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from tools.v22_static_reuse import StaticGraphs, FrozenFields, ReuseSaver, installed_model, installed_reference
from tools.v22_artifacts import tree_hash
from tools.v22_rank_objective import RankingContext
from tools.benchmark_v22_recompute_debug import exact, frozen


class ReuseTests(unittest.TestCase):
    def test_plan_validation_is_repeated_after_inplace_mutation(self):
        cache = StaticGraphs()
        values = (torch.randn(12, 128), torch.arange(12) % 3, torch.arange(12) % 2)
        plan = dict(zip(('support_embeddings', 'owners', 'classes'), (x.clone() for x in values)))
        cache.check_plan(plan, *values)
        with patch('tools.v22_static_reuse.torch.equal', wraps=torch.equal) as equal:
            cache.check_plan(plan, *values)
            self.assertEqual(equal.call_count, 0)
        plan['support_embeddings'].add_(1)
        with self.assertRaisesRegex(ValueError, 'different support'):
            cache.check_plan(plan, *values)

    def test_topology_invalidates_on_mutation_and_replacement(self):
        cache = StaticGraphs()
        x = torch.randn(12, 128); owners = torch.arange(12) % 3; labels = torch.arange(12) % 2
        first = cache.topology(x, owners, labels)
        self.assertIs(first, cache.topology(x + 1, owners, labels))
        labels[0] = 1
        self.assertIsNot(first, cache.topology(x, owners, labels))
        self.assertEqual(cache.support_builds, 2)
        cache.topology(x, owners.clone(), labels)
        self.assertEqual(cache.support_builds, 3)
        owners[0] = -1
        with self.assertRaises(RuntimeError):
            cache.topology(x, owners, labels)

    def test_reference_reuses_complete_case_and_invalidates_memory(self):
        data = SimpleNamespace(rows=[dict(id=str(i), case_id='A' if i < 4 else 'B', target=i % 2) for i in range(8)])
        memory = dict(record_ids=[str(i) for i in range(8)], embeddings=torch.randn(8, 128))
        context = RankingContext(data, memory)
        expected = [context.reference(ids, 'cpu') for ids in ([0, 1], [2, 3], [4, 5])]
        with installed_reference(context) as counts:
            actual = [context.reference(ids, 'cpu') for ids in ([0, 1], [2, 3], [4, 5])]
            self.assertTrue(exact(expected, actual))
            self.assertIs(actual[0][0], actual[1][0])
            self.assertEqual(counts['reference_builds'], 2)
            memory['embeddings'].add_(1)
            changed = context.reference([4, 5], 'cpu')
            self.assertTrue(torch.equal(changed[0], actual[2][0] + 1))
            with self.assertRaisesRegex(ValueError, 'Duplicated'):
                context.reference([0, 0], 'cpu')

    def test_frozen_cache_tracks_nested_mutations_and_keeps_old_snapshot(self):
        cache = FrozenFields()
        value = dict(x=torch.ones(4, 128), ids=['a', 'b'], gen=dict(epoch=0))
        old, digest = cache.get('memory', value)
        self.assertIs(cache.get('memory', value)[0], old)
        value['x'][0] += 2
        new, current = cache.get('memory', value)
        self.assertNotEqual(current, digest)
        self.assertTrue(torch.equal(old['x'], torch.ones(4, 128)))
        value['ids'][0] = 'c'
        self.assertNotEqual(cache.get('memory', value)[1], current)
        value['gen']['epoch'] = 1
        self.assertEqual(cache.get('memory', value)[0]['gen']['epoch'], 1)
        self.assertEqual(cache.misses, 4)

    def test_saver_preserves_full_hashes_current_weights_and_durable_cursor(self):
        from tools.v22_resume_integrity import seal_resume
        with tempfile.TemporaryDirectory() as folder:
            net = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128))
            opt = torch.optim.AdamW(net.parameters())
            net(torch.randn(4, 128)).sum().backward(); opt.step()
            saver = ReuseSaver(Path(folder), net, opt, dict(debug=True, artifact_contract='observed_rank_artifact_v5'))
            state = dict(phase='optimization', epoch=0, step=1,
                         memory=dict(embeddings=torch.randn(11279, 128), record_ids=['a']),
                         memory_generation={'epoch': 0}, plan={'x': torch.randn(5, 128)},
                         plan_generation={'step': 0}, memory_work=None)
            try:
                saver.save(state); saver.flush()
                prior = torch.load(Path(folder)/'checkpoint_latest.pt', weights_only=False)
                with torch.no_grad():
                    net[0].weight.add_(.25); net[1].running_mean.add_(1)
                state['step'] = 2
                saver.save(state); saver.flush()
                self.assertEqual(saver.last_reuse_metrics['reused_fields'], 4)
                current = torch.load(Path(folder)/'checkpoint_latest.pt', weights_only=False)
                reference = copy.deepcopy(current); seal_resume(reference)
                for key in ('model_sha256', 'rng_sha256', 'resume_integrity'):
                    self.assertEqual(current[key], reference[key])
                self.assertNotEqual(prior['model_sha256'], current['model_sha256'])
                self.assertEqual(current['model_sha256'], tree_hash(net.state_dict()))
                self.assertEqual(saver.committed['step'], 2)
                state['memory']['embeddings'].add_(1)
                saver.save(state); saver.flush()
                changed = torch.load(Path(folder)/'checkpoint_latest.pt', weights_only=False)
                self.assertEqual(changed['resume_integrity']['memory'], tree_hash(state['memory']))
                self.assertNotEqual(prior['resume_integrity']['memory'], changed['resume_integrity']['memory'])
            finally:
                saver.close()

    def test_fixed_support_and_growing_prefix_remain_fresh(self):
        with tempfile.TemporaryDirectory() as folder:
            net = torch.nn.Linear(128, 128); opt = torch.optim.AdamW(net.parameters())
            saver = ReuseSaver(Path(folder), net, opt, dict(debug=True, artifact_contract='observed_rank_artifact_v5'))
            state = dict(phase='refresh_memory', epoch=0, step=0, memory=None, plan=None)
            try:
                with saver.fixed_support():
                    for n in (4, 8):
                        state.update(memory_work=torch.full((n, 128), float(n)), memory_next=n)
                        saver.save(state); saver.flush()
                        saved = torch.load(Path(folder)/'checkpoint_latest.pt', weights_only=False)
                        self.assertEqual(saved['state']['memory_work'].shape[0], n)
                        self.assertEqual(saved['resume_integrity']['memory_work'], tree_hash(state['memory_work']))
                    with torch.no_grad(): net.weight.add_(1)
                    with self.assertRaisesRegex(RuntimeError, 'Model changed'):
                        saver.save(state)
            finally:
                saver.close()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    def test_full_support_cuda_two_updates_and_evaluation_exact(self):
        from hiercp_v222.model import PromptGraphModel
        from hiercp_v222.v1_cache import configuration
        from hiercp_v222.v1_execution import rng_state, restore_rng
        cfg, base = configuration()
        old_threads = torch.get_num_threads(); torch.set_num_threads(4)
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(42)
            # All production L1/L2 widths/depths and full 11279 support count.
            # L0 is a labelled synthetic linear fixture; it is not a CT benchmark.
            model = PromptGraphModel(cfg, base, {}, local_encoder=torch.nn.Linear(128, 128)).cuda()
            support = (torch.randn(11279, 128, device='cuda'), torch.arange(11279, device='cuda') % 90,
                       (torch.arange(11279, device='cuda') // 90) % 2)
            with torch.autocast('cuda', dtype=torch.bfloat16): plan = model.fit_support_clusters(*support)
            queries = [torch.randn(128, 128, device='cuda') for _ in range(2)]
            initial = frozen(model.state_dict()); random = rng_state(); runs = []
            for optimized in (False, True):
                model.load_state_dict(initial); model.train(); restore_rng(random)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                evidence = []
                with installed_model(model) if optimized else nullcontext():
                    for query in queries:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            output = model.predict_embeddings(model.local(query), model.prepare_support(*support, cluster_plan=plan))
                            loss = output['logits'].float().square().mean() + output['alignment_loss']
                        loss.backward()
                        gradients = {k: p.grad for k, p in model.named_parameters()}
                        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values()))
                        gradients = frozen(gradients)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        evidence.append(frozen(dict(loss=loss, gradients=gradients, model=model.state_dict(),
                                                    adam=optimizer.state_dict(), rng=rng_state())))
                    model.eval()
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        evidence.append(frozen(model.predict_embeddings(model.local(queries[0]),
                                               model.prepare_support(*support, cluster_plan=plan))))
                runs.append(evidence)
            self.assertTrue(exact(runs[0], runs[1]))
        finally:
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_deterministic)


if __name__ == '__main__':
    unittest.main()
