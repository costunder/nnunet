import ast
import hashlib

def _method(source, class_name, method):
    classes = [node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name]
    methods = [] if len(classes) != 1 else [node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method]
    if len(methods) != 1:
        raise ValueError('Exactly one actual private method required: ' + method)
    return methods[0]

def prove_readonly_dataset_source(source):
    """Admit the installed Blosc2 load/unpack path, never its save functions."""
    unpack = _method(source, 'nnUNetDatasetBlosc2', 'unpack_dataset')
    body = unpack.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Pass):
        raise ValueError('Actual Blosc2 unpack must be a pure no-op before preprocessing reuse')
    load = _method(source, 'nnUNetDatasetBlosc2', 'load_case')
    calls = [node for node in ast.walk(load) if isinstance(node, ast.Call)]
    opens = [node for node in calls if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'blosc2' and node.func.attr == 'open']
    if len(opens) != 3 or any(not any(keyword.arg == 'mode' and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 'r' for keyword in node.keywords) for node in opens):
        raise ValueError('All actual data/segmentation/previous-stage Blosc2 opens must be read-only')
    constructors = _method(source, 'nnUNetDatasetBlosc2', '__init__')
    mmap = [node for node in ast.walk(constructors) if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == 'mmap_mode' for key in node.keys)]
    if len(mmap) != 1 or any(value.value != 'r' for key, value in zip(mmap[0].keys, mmap[0].values)
            if isinstance(key, ast.Constant) and key.value == 'mmap_mode'):
        raise ValueError('Actual Linux Blosc2 dataset must use read-only mmap')
    forbidden = {'save_case', 'save_seg', 'remove', 'unlink', 'asarray', 'write_pickle'}
    if any((isinstance(node.func, ast.Name) and node.func.id in forbidden)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in forbidden) for node in calls):
        raise ValueError('Dataset read path unexpectedly writes preprocessing')
    return dict(dataset_class='nnUNetDatasetBlosc2', unpack_is_noop=True,
        data_seg_previous_stage_mode='r', Linux_mmap_mode='r',
        original_preprocessing_reused_without_copy=True,
        unpack_AST_sha256=hashlib.sha256(ast.dump(unpack).encode()).hexdigest(),
        load_case_AST_sha256=hashlib.sha256(ast.dump(load).encode()).hexdigest())
