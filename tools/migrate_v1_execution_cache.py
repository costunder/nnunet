"""New cache manifest for reviewed execution-only changes; no graph rewrite."""
import ast
import copy
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hiercp_v222.v1_cache import provenance
from hiercp_v222.contracts import read_json,write_new,sha

def structural_local(source):
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='copy')]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocalBatch')
    cls.body=[n for n in cls.body if not (isinstance(n,ast.FunctionDef) and n.name=='to')]
    return ast.dump(tree,include_attributes=False)

def migrate(original,destination):
    original=Path(original).resolve();destination=Path(destination).resolve()
    if destination.parent!=original.parent:raise ValueError('Manifest must retain the original immutable cache root')
    meta=read_json(original);current=provenance();old=meta['source_identity']
    if not meta['complete'] or meta['debug']:raise ValueError('Complete real cohort required')
    archive=ROOT/'versions/v2.22/v1_full_training_empty_context_20260924/source.zip'
    local='hiercp_v222/v1_local.py'
    with zipfile.ZipFile(archive) as z:
        data=z.read(local)
        old_training=z.read('hiercp_v222/v1_training.py').decode('utf-8-sig')
    import hashlib
    if hashlib.sha256(data).hexdigest()!=old[local]:raise ValueError('Archived local source does not match original cache')
    if structural_local(data.decode('utf-8-sig'))!=structural_local((ROOT/local).read_text(encoding='utf-8-sig')):
        raise ValueError('Local architecture/geometry changed beyond reviewed transfer method')
    def helpers(source):
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='train')]
        # Evaluation now releases completed GPU outputs and invokes an optional
        # allocator callback. Strip exactly those statements; metrics stay identical.
        class ExecutionOnly(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                if node.name=='evaluate' and node.args.args[-1].arg=='batch_complete':
                    node.args.args.pop();node.args.defaults.pop()
                    return self.generic_visit(node)
                return node
            def visit_Delete(self,node):
                return None if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='result' else node
            def visit_If(self,node):
                if ast.dump(node.test)==ast.dump(ast.parse('batch_complete is not None',mode='eval').body):
                    if ast.dump(node.body[0])!=ast.dump(ast.parse('batch_complete()').body[0]) or len(node.body)!=1 or node.orelse:
                        raise ValueError('Unexpected evaluation execution callback')
                    return None
                return self.generic_visit(node)
        tree=ExecutionOnly().visit(tree)
        return ast.dump(tree,include_attributes=False)
    if helpers(old_training)!=helpers((ROOT/'hiercp_v222/v1_training.py').read_text(encoding='utf-8-sig')):
        raise ValueError('Sampling, calibration or evaluation helpers changed')
    changed=sorted(k for k in set(old)|set(current) if old.get(k)!=current.get(k))
    allowed={local,'hiercp_v222/v1_training.py','hiercp_v222/v1_execution.py','run_v222_v1_l0.py'}
    if set(changed)-allowed:raise ValueError(f'Unreviewed cache construction change: {set(changed)-allowed}')
    if len(meta['records'])!=14102:raise ValueError('Unexpected cohort size')
    # Per-record SHA verification stays enabled in PairDataset.record().
    updated=copy.deepcopy(meta);updated['source_identity']=current
    updated['execution_migration']=dict(original=str(original),original_sha256=sha(original),
        changed_files=changed,local_AST_identical_except_transfer=True,graph_files_rewritten=0,
        graph_count=len(meta['records']),original_source_identity=old)
    write_new(destination,updated)
    print(json.dumps({k:v for k,v in updated['execution_migration'].items() if k!='original_source_identity'},indent=2))
    return destination

if __name__=='__main__':migrate(*sys.argv[1:])
