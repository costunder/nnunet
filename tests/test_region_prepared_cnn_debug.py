"""Synthetic CNN receipt tests; not trained features or a production cache."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from hiercp_v222.v1_cache import configuration
from hiercp_v222.v1_local import V1LocalEncoder
from tools.v222_review_contracts import installed
from tools.v22_artifacts import tree_hash
from l0_regions.training_data import reference_from_prepared,source_identity,prepare_cache,FORMAT


class PreparedCNN(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        cfg,base=configuration()
        with installed('stride4'):self.net=V1LocalEncoder(base)
        self.weights=copy.deepcopy(self.net.dense_encoder.state_dict())
        torch.save(self.weights,self.root/'frozen_cnn.pt')
        self.meta=dict(format=FORMAT,debug=True,config=cfg,base=base,source_identity=source_identity(preparation=True),
            cnn_sha256=tree_hash(self.weights),cnn_file_sha256=self.filehash(),partition_checkpoint_sha256='synthetic_test_only')
        self.save()
    def tearDown(self):self.tmp.cleanup()
    def filehash(self):return hashlib.sha256((self.root/'frozen_cnn.pt').read_bytes()).hexdigest()
    def save(self):(self.root/'request.json').write_text(json.dumps(self.meta))
    def test_exact_cnn_without_parent_load(self):
        with patch('l0_regions.training_data.reference_from_checkpoint',side_effect=OSError(116,'Stale file handle')) as parent:
            net,origin,digest=reference_from_prepared(self.root,True)
        self.assertEqual(parent.call_count,0)
        self.assertEqual(tree_hash(net.dense_encoder.state_dict()),tree_hash(self.weights))
        self.assertEqual(origin,self.meta)
        self.assertEqual(digest,hashlib.sha256((self.root/'request.json').read_bytes()).hexdigest())
    def test_debug_not_promoted(self):
        with self.assertRaisesRegex(ValueError,'mode'):reference_from_prepared(self.root,False)
    def test_corrupt_file_rejected(self):
        with (self.root/'frozen_cnn.pt').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'file changed'):reference_from_prepared(self.root,True)
    def test_changed_tensor_rejected_even_with_new_file_digest(self):
        next(iter(self.weights.values())).add_(1)
        torch.save(self.weights,self.root/'frozen_cnn.pt')
        self.meta['cnn_file_sha256']=self.filehash();self.save()
        with self.assertRaisesRegex(ValueError,'contents differ'):reference_from_prepared(self.root,True)
    def test_changed_architecture_rejected(self):
        self.weights[next(iter(self.weights))]=torch.zeros(1)
        torch.save(self.weights,self.root/'frozen_cnn.pt')
        self.meta.update(cnn_file_sha256=self.filehash(),cnn_sha256=tree_hash(self.weights));self.save()
        with self.assertRaises(RuntimeError):reference_from_prepared(self.root,True)
    def test_changed_source_rejected(self):
        self.meta['source_identity']['core']['config/train.json']='changed';self.save()
        with self.assertRaisesRegex(ValueError,'core implementation'):reference_from_prepared(self.root,True)
    def test_ambiguous_explicit_parent_not_ignored(self):
        with self.assertRaisesRegex(ValueError,'omit --partition-checkpoint'):
            prepare_cache('index.json','parent.pt',self.root/'new',batch=32,workers=16,reg1=.02,view_epoch=0,
                          budget=None,reuse_prepared=self.root)
        self.assertFalse((self.root/'new').exists())


if __name__=='__main__':unittest.main(verbosity=2)
