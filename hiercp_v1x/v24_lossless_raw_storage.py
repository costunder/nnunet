"""Explicit v24 lossless raw-bank storage; legacy NPY defaults stay intact.

Only the two complete native baseline volumes use chunked Blosc2. The CT
remains float64 and labels int16. Every voxel is compared byte-for-byte before
publication; runtime returns only requested crops and never decodes a whole
volume as an implicit fallback. Original donor NPZ/NPY payloads remain valid.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.util
from itertools import product
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import weakref

import numpy as np

ORIGINAL_SOURCE_PATH=Path(__file__).resolve().parents[1]/'custom_trainers/onlinecp_raw_bank.py'
ORIGINAL_SOURCE_SHA256=hashlib.sha256(ORIGINAL_SOURCE_PATH.read_bytes()).hexdigest()
_backend_spec=importlib.util.spec_from_file_location(
    '_v24_lossless_raw_backend_'+ORIGINAL_SOURCE_SHA256,ORIGINAL_SOURCE_PATH)
original=importlib.util.module_from_spec(_backend_spec)
_backend_spec.loader.exec_module(original)
if (Path(original.__file__).resolve()!=ORIGINAL_SOURCE_PATH
        or hashlib.sha256(ORIGINAL_SOURCE_PATH.read_bytes()).hexdigest()!=ORIGINAL_SOURCE_SHA256):
    raise ValueError('Exact ROOT raw-bank implementation changed during activation')


FORMAT='v24_blosc2_lossless_v1'
CHUNKS=(1,32,128,128)
BLOCKS_FLOAT64=(1,1,128,128)
BLOCKS_INT16=(1,4,128,128)
COMPRESSION=dict(codec='ZSTD',clevel=5,filter='BITSHUFFLE',nthreads=1,
                 lossy_filters=False,original_dtype_preserved=True)


def _blosc():
    import blosc2
    return blosc2


def _slices(shape,chunks):
    for starts in product(*(range(0,n,c) for n,c in zip(shape,chunks))):
        yield tuple(slice(start,min(start+chunk,n))
                    for start,chunk,n in zip(starts,chunks,shape))


def _partition(shape,requested):
    return tuple(min(n,c) for n,c in zip(shape,requested))


def _open(path):
    b=_blosc()
    # Regular logical handles use per-operation I/O, not a patient-sized mmap.
    # Do not override cparams on open: Blosc2 would replace the reported
    # original filter header with defaults, defeating its integrity check.
    # Only decompression occurs here, and its thread count is explicitly one.
    return b.open(str(path),mode='r',dparams={'nthreads':1})


def _validate_decoder(decoder,spec):
    b=_blosc();c=decoder.cparams
    if (tuple(decoder.shape)!=tuple(spec['shape']) or decoder.dtype.str!=spec['dtype']
            or tuple(decoder.chunks)!=tuple(spec['chunks'])
            or tuple(decoder.blocks)!=tuple(spec['blocks'])
            or c.codec!=b.Codec.ZSTD or c.clevel!=5
            or list(c.filters)!=[b.Filter.NOFILTER]*5+[b.Filter.BITSHUFFLE]
            or any(c.filters_meta) or c.codec_meta!=0
            or decoder.dparams.nthreads!=1):
        raise ValueError('Lossless Blosc2 shape/dtype/chunk/compression contract changed')


def _verify_all(array,path,spec):
    decoder=_open(path)
    try:
        _validate_decoder(decoder,spec)
        source_hash=hashlib.sha256();decoded_hash=hashlib.sha256();count=0;tiles=0
        for index in _slices(array.shape,spec['chunks']):
            expected=np.ascontiguousarray(array[index])
            restored=decoder[index]
            if (restored.dtype!=expected.dtype or restored.shape!=expected.shape
                    or restored.tobytes(order='C')!=expected.tobytes(order='C')):
                raise ValueError('Lossless storage failed an original full-precision voxel roundtrip')
            source_hash.update(expected.tobytes(order='C'));decoded_hash.update(restored.tobytes(order='C'))
            count+=expected.size;tiles+=1
        if count!=array.size or source_hash.digest()!=decoded_hash.digest():
            raise ValueError('Lossless roundtrip did not cover every original voxel')
        return dict(all_voxels_verified=True,verified_voxels=count,verification_tiles=tiles,
                    source_tile_bytes_sha256=source_hash.hexdigest(),
                    decoded_tile_bytes_sha256=decoded_hash.hexdigest(),
                    traversal='lexicographic full chunk tiles in C order',
                    maximum_verification_tile_bytes=int(np.prod(spec['chunks']))*array.dtype.itemsize)
    finally:
        # Blosc2 4.3 regular NDArray handles have no close(); releasing their
        # last reference releases the logical decoder. No mmap is opened.
        decoder=None


def _save_volume(root,relative,array):
    path=original._relative_path(root,relative,must_exist=False)
    spec=dict(shape=list(array.shape),dtype=array.dtype.str,storage=FORMAT,
              chunks=list(_partition(array.shape,CHUNKS)),
              blocks=list(_partition(array.shape,BLOCKS_FLOAT64 if array.dtype.itemsize==8 else BLOCKS_INT16)),
              compression=dict(COMPRESSION))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        original._relative_path(root,relative)
        proof=_verify_all(array,path,spec)
    else:
        descriptor,name=tempfile.mkstemp(prefix=path.name+'.pending.',suffix='.b2nd',dir=path.parent)
        os.close(descriptor);temporary=Path(name)
        try:
            b=_blosc()
            decoder=b.empty(array.shape,dtype=array.dtype,urlpath=str(temporary),mode='w',
                chunks=spec['chunks'],blocks=spec['blocks'],
                cparams=dict(codec=b.Codec.ZSTD,clevel=5,nthreads=1,
                             filters=[b.Filter.NOFILTER]*5+[b.Filter.BITSHUFFLE]),
                dparams=dict(nthreads=1))
            for index in _slices(array.shape,spec['chunks']):
                decoder[index]=np.ascontiguousarray(array[index])
            decoder=None
            proof=_verify_all(array,temporary,spec)
            with temporary.open('r+b') as handle:handle.flush();os.fsync(handle.fileno())
            try:
                os.link(temporary,path)
            except FileExistsError:
                # Concurrent immutable publication is accepted only after the
                # existing file's entire original-value roundtrip is checked.
                original._relative_path(root,relative)
                proof=_verify_all(array,path,spec)
        finally:
            temporary.unlink(missing_ok=True)
    spec.update(path=relative,sha256=original._sha(path),roundtrip=proof,
                uncompressed_bytes=int(array.nbytes),stored_bytes=path.stat().st_size)
    return spec


def save_case(bank_root,relative_manifest,case):
    """Publish all original case data; only complete baseline volumes compress."""
    value={key:value for key,value in case.items() if key!='preparation'}
    for name,dtype in (('baseline_unclipped',np.dtype('float64')),('baseline_seg',np.dtype('int16'))):
        array=value.get(name)
        if (not isinstance(array,np.ndarray) or array.dtype!=dtype
                or array.ndim!=4 or array.shape[0]!=1 or any(n<=0 for n in array.shape)):
            raise ValueError('Original one-channel float64 CT and int16 labels required: '+name)
    if value['baseline_unclipped'].shape!=value['baseline_seg'].shape:
        raise ValueError('Complete CT and label grids differ')
    arrays={};tree=original._encode_tree(value,arrays)
    compressed={tree[name]['$array'] for name in ('baseline_unclipped','baseline_seg')}
    manifest=dict(format=original.STORAGE_FORMAT,kind='case',tree=tree,arrays={},
                  v24_storage=FORMAT,compression=dict(COMPRESSION))
    for key,array in arrays.items():
        suffix='.b2nd' if key in compressed else '.npy'
        relative=str(PurePosixPath(relative_manifest).with_suffix(''))+'.'+key+suffix
        if key in compressed:
            spec=_save_volume(bank_root,relative,array)
        else:
            path=original._relative_path(bank_root,relative,must_exist=False)
            checksum=original._publish(path,lambda handle,a=array:np.save(handle,a,allow_pickle=False))
            spec=dict(shape=list(array.shape),dtype=array.dtype.str,path=relative,sha256=checksum)
        manifest['arrays'][key]=spec
    volumes={name:manifest['arrays'][tree[name]['$array']]
             for name in ('baseline_unclipped','baseline_seg')}
    manifest['lossless_storage']=dict(format=FORMAT,volume_count=2,
        original_logical_array_bytes=sum(int(array.nbytes) for array in arrays.values()),
        original_volume_bytes=sum(spec['uncompressed_bytes'] for spec in volumes.values()),
        compressed_volume_file_bytes=sum(spec['stored_bytes'] for spec in volumes.values()),
        full_precision_volume_roundtrip_verified=True,
        verified_voxels=sum(spec['roundtrip']['verified_voxels'] for spec in volumes.values()),
        volumes={name:dict(dtype=spec['dtype'],shape=spec['shape'],
            uncompressed_bytes=spec['uncompressed_bytes'],stored_bytes=spec['stored_bytes'],
            source_tile_bytes_sha256=spec['roundtrip']['source_tile_bytes_sha256'],
            decoded_tile_bytes_sha256=spec['roundtrip']['decoded_tile_bytes_sha256'],
            verified_voxels=spec['roundtrip']['verified_voxels']) for name,spec in volumes.items()},
        candidate_storage_unchanged=True,whole_volume_runtime_decode=False,
        compression=dict(COMPRESSION))
    encoded=(json.dumps(manifest,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode('utf8')
    return original._publish(original._relative_path(bank_root,relative_manifest,must_exist=False),
                             lambda handle:handle.write(encoded))


class _CropArray:
    """Read-only exact-dtype crop interface; no implicit full-array conversion."""
    def __init__(self,store,spec):
        self._store=weakref.ref(store);self._spec=dict(spec);self._decoder=None;self._closed=False
        self.shape=tuple(spec['shape']);self.dtype=np.dtype(spec['dtype']);self.ndim=len(self.shape)
        self.size=int(np.prod(self.shape));self.nbytes=self.size*self.dtype.itemsize

    def __getitem__(self,index):
        if self._closed:raise ValueError('Lossless crop array is closed')
        store=self._store()
        if store is None:raise ValueError('Lossless crop storage owner is unavailable')
        path=store._check(self._spec['path'],self._spec['sha256'])
        if not isinstance(index,tuple):index=(index,)
        if any(item is Ellipsis for item in index):
            if sum(item is Ellipsis for item in index)!=1:raise IndexError('Only one ellipsis is allowed')
            expanded=[]
            for item in index:
                expanded.extend([slice(None)]*(self.ndim-len(index)+1)) if item is Ellipsis else expanded.append(item)
            index=tuple(expanded)
        if len(index)>self.ndim:raise IndexError('Too many crop axes')
        index=(*index,*([slice(None)]*(self.ndim-len(index))))
        if any(not isinstance(item,(slice,int,np.integer))
               or (isinstance(item,slice) and item.step not in (None,1)) for item in index):
            raise IndexError('Lossless native crop requires basic contiguous slices or integer axes')
        if self._decoder is None:
            self._decoder=_open(path);_validate_decoder(self._decoder,self._spec)
        output=self._decoder[index]
        output.flags.writeable=False
        return output

    def __array__(self,dtype=None,copy=None):
        raise TypeError('Use explicit crop slices; implicit whole-volume decoding is forbidden')

    def close(self):
        self._decoder=None;self._closed=True


class LosslessRawBankStore(original.RawBankStore):
    """Original candidate storage plus lossless, lazily decoded case crops."""
    def __init__(self,bank_root):
        super().__init__(bank_root);self._proxies=weakref.WeakSet()

    def __getstate__(self):
        return {**super().__getstate__(),'_proxies':None}

    def close(self):
        if self._proxies is not None:
            for proxy in tuple(self._proxies):proxy.close()
            self._proxies.clear()
        super().close()

    def load_case(self,relative,sha256):
        path=self._check(relative,sha256);manifest=json.loads(path.read_text(encoding='utf8'))
        if manifest.get('v24_storage') is None:
            return super().load_case(relative,sha256)
        if (manifest.get('format')!=original.STORAGE_FORMAT or manifest.get('kind')!='case'
                or manifest.get('v24_storage')!=FORMAT or manifest.get('compression')!=COMPRESSION):
            raise ValueError('Incompatible explicit lossless case manifest')
        specs=manifest['arrays'];key=(str(relative),sha256);cached=self._cases.get(key);arrays={}
        if self._proxies is None:self._proxies=weakref.WeakSet()
        for name,spec in specs.items():
            array_path=self._check(spec['path'],spec['sha256'])
            if spec.get('storage')==FORMAT:
                if (spec.get('compression')!=COMPRESSION or spec.get('roundtrip',{}).get('all_voxels_verified') is not True
                        or np.dtype(spec['dtype']) not in (np.dtype('float64'),np.dtype('int16'))
                        or len(spec['shape'])!=4 or spec['shape'][0]!=1
                        or spec['roundtrip'].get('verified_voxels')!=int(np.prod(spec['shape']))
                        or spec['roundtrip'].get('source_tile_bytes_sha256')!=spec['roundtrip'].get('decoded_tile_bytes_sha256')):
                    raise ValueError('Incomplete original-precision roundtrip proof')
                arrays[name]=cached[0][name] if cached is not None else _CropArray(self,spec)
                self._proxies.add(arrays[name])
            elif 'storage' in spec:
                raise ValueError('Unknown raw case storage; no decoding fallback')
            else:
                arrays[name]=cached[0][name] if cached is not None else np.load(array_path,mmap_mode='r',allow_pickle=False)
                if list(arrays[name].shape)!=spec['shape'] or arrays[name].dtype.str!=spec['dtype']:
                    raise ValueError('Raw-bank array shape/dtype mismatch')
                arrays[name].flags.writeable=False
        referenced=set()
        def decode(value):
            if isinstance(value,dict):
                if '$array' in value:
                    if set(value)!={'$array'} or value['$array'] not in arrays:raise ValueError('Invalid lossless array reference')
                    referenced.add(value['$array']);return arrays[value['$array']]
                return {name:decode(item) for name,item in value.items()}
            if isinstance(value,list):return [decode(item) for item in value]
            return value
        result=decode(manifest['tree']);decode=None
        if referenced!=set(arrays) or not isinstance(result,dict):raise ValueError('Unused arrays or malformed case root')
        for name,dtype in (('baseline_unclipped',np.dtype('float64')),('baseline_seg',np.dtype('int16'))):
            if not isinstance(result.get(name),_CropArray) or result[name].dtype!=dtype:
                raise ValueError('Lossless full-precision CT/label roles changed')
        self._cases[key]=(arrays,result);self._cases.move_to_end(key)
        while len(self._cases)>8:self._cases.popitem(last=False)
        return result
