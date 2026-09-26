"""DEBUG numeric operator tests; synthetic graphs are not medical results."""
import unittest
import torch
from hiercp.model import _StreamedEdgeAggregation
from tools.v222_support_sparse_debug import SparseEvaluation


class SparseSupportDebugTest(unittest.TestCase):
    def test_multigraph_and_empty_graph_match_double_precision(self):
        torch.manual_seed(423)
        features = torch.randn(7, 4, 8, dtype=torch.float64)
        for edge_count in (0, 73):
            edges = torch.stack((torch.arange(edge_count)%7, torch.arange(edge_count)%5))
            weights = torch.rand(edge_count, 4, dtype=torch.float64)
            operator = SparseEvaluation()
            with torch.no_grad():
                expected = _StreamedEdgeAggregation.apply(features, edges, weights, 6, 13)
                actual = operator.apply(features, edges, weights, 6, 13)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                # Topology may be reused, learned attention must not be reused.
                second = operator.apply(features, edges, weights*2, 6, 13)
                torch.testing.assert_close(second, actual*2, rtol=1e-12, atol=1e-12)
                self.assertEqual(len(operator.topologies), 1)

    def test_cannot_silently_replace_training(self):
        with self.assertRaisesRegex(RuntimeError, 'evaluation-only'):
            SparseEvaluation().apply(torch.randn(3,2,4), torch.zeros(2,1,dtype=torch.long),
                torch.ones(1,2), 3, 10)


if __name__=='__main__':unittest.main()
