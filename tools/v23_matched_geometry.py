"""Exact CPU geometry acceleration; no candidate, lesion or graph is removed."""
from types import SimpleNamespace
import numpy as np
from scipy import ndimage as ndi


def distance_at_voxel(mask, spacing, center, pool):
    """The same float32 Euclidean EDT value, evaluated only where consumed.

    Visit every occupied voxel in bounded chunks; four independent reductions
    run on the caller's existing pool, without a volume-sized distance field.
    """
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return np.float32(np.inf)
    spacing = np.asarray(spacing, dtype=np.float32).astype(np.float64)
    center = np.asarray(center, dtype=np.int64)
    edges = np.linspace(0, len(coordinates), 5, dtype=np.int64)

    def minimum(bounds):
        first, stop = bounds
        best = np.inf
        for start in range(int(first), int(stop), 262144):
            delta = (coordinates[start:min(start + 262144, stop)] - center) * spacing
            squared = np.sum(delta * delta, axis=1)
            best = min(best, float(squared.min()))
        return best

    minimum_squared = min(pool.map(minimum, zip(edges[:-1], edges[1:])))
    return np.float32(np.sqrt(minimum_squared))


class PatientGeometry:
    """Case-scoped immutable data and one cached recipient descriptor table."""
    def __init__(self, case, regions, collection, pool):
        if case.label.flags.writeable or case.image.flags.writeable:
            raise ValueError('Immutable patient arrays required for exact cached geometry')
        self.case, self.regions, self.collection, self.pool = case, regions, collection, pool
        self.boxes = ndi.find_objects(collection.components)
        self.lesions = {}
        self.liver = {}
        self.statistics = dict(recipient_builds=0, recipient_cache_hits=0, source_distance_queries=0)

    def recipient_lesions(self, case, regions, *, hierarchy, schema, tumor_label, max_lesions, ct_clip):
        if case is not self.case or regions is not self.regions or tumor_label != 2:
            raise ValueError('Cached recipient data belongs to a different actual patient')
        count = len(self.collection.sizes) - 1
        if max_lesions is not None and int(max_lesions) > 0 and count > int(max_lesions):
            raise RuntimeError('Full lesion count exceeds original contract; no lesion discarded')
        key = (tuple(ct_clip), max_lesions)
        if key in self.lesions:
            self.statistics['recipient_cache_hits'] += 1
            return self.lesions[key]
        components, sizes = self.collection.components, self.collection.sizes
        component_ids = sorted(range(1, count + 1), key=lambda value: int(sizes[value]), reverse=True)

        def descriptor(component):
            box = self.boxes[component - 1]
            mask = components[box] == component
            offset = np.asarray([s.start for s in box])
            centroid = np.asarray(ndi.center_of_mass(mask)) + offset
            center = tuple(int(np.clip(round(value), 0, case.shape[axis] - 1)) for axis, value in enumerate(centroid))
            context_box = tuple(slice(max(0, s.start - 3), min(case.shape[axis], s.stop + 3)) for axis, s in enumerate(box))
            # The archived context function first takes exactly this padded box.
            from hiercp.common import context_stats_for_local_mask
            mean, std = context_stats_for_local_mask(case.image[context_box], regions.full_organ_mask[context_box],
                components[context_box] == component, ring_width=2)
            depth = float(regions.organ_depth[center])
            row = hierarchy.upper_geometry_vector(center=center, shape=case.shape,
                border_distance_mm=depth, occupied_distance_mm=0., context_mean_hu=mean, context_std_hu=std,
                volume_vox=int(sizes[component]), coverage=1., local_thickness_mm=max(2. * depth, float(np.min(case.spacing))),
                surface_alignment=0., scale_mean=1., anisotropy=1., valid=1., ct_clip=ct_clip)
            return row, hierarchy.normalized_position(center, case.shape).astype(np.float32), regions.region_at(center)

        rows = list(self.pool.map(descriptor, component_ids))
        if rows:
            raw, positions, region_ids = zip(*rows)
            value = np.stack(raw).astype(np.float32), np.stack(positions).astype(np.float32), np.asarray(region_ids, np.int64), component_ids
        else:
            value = np.empty((0, schema.UPPER_RAW_DIM), np.float32), np.empty((0, 3), np.float32), np.empty((0,), np.int64), component_ids
        self.lesions[key] = value
        self.statistics['recipient_builds'] += 1
        return value

    def runtime(self, original_runtime):
        runtime, hashes = original_runtime()
        original = runtime['hierarchy']
        proxy = SimpleNamespace(**vars(original))

        def source_raw(case, source, regions, occupied_without_source, *, ct_clip):
            mean, std = original._context_stats(case, source.full_mask, regions.full_organ_mask)
            distance = distance_at_voxel(occupied_without_source, case.spacing, source.anchor_center, self.pool)
            self.statistics['source_distance_queries'] += 1
            axis, anisotropy = original._principal_axis(source.full_mask, case.spacing)
            normal = original._surface_normal(regions.organ_depth, source.anchor_center, case.spacing)
            depth = float(regions.organ_depth[source.anchor_center])
            return original.upper_geometry_vector(center=source.anchor_center, shape=case.shape,
                border_distance_mm=depth, occupied_distance_mm=float(distance), context_mean_hu=mean, context_std_hu=std,
                volume_vox=source.voxel_count, coverage=1., local_thickness_mm=max(2. * depth, float(np.min(case.spacing))),
                surface_alignment=float(abs(np.dot(axis, normal))), scale_mean=1., anisotropy=anisotropy, ct_clip=ct_clip)

        def liver_raw(case, regions, *, tumor_label, ct_clip):
            if case is not self.case or regions is not self.regions:
                raise ValueError('Cached liver descriptors belong to another actual patient')
            key = (tumor_label, tuple(ct_clip))
            if key not in self.liver:
                self.liver[key] = original._liver_raw(case, regions, tumor_label=tumor_label, ct_clip=ct_clip)
            return self.liver[key]

        proxy._source_raw, proxy._liver_raw = source_raw, liver_raw
        return dict(runtime, hierarchy=proxy), hashes

    def bind(self, upper):
        # 'upper' is already a private clone; archived module functions stay intact.
        original_runtime = upper.__globals__['_runtime']
        upper.__globals__['_runtime'] = lambda: self.runtime(original_runtime)
        upper.__globals__['_recipient_lesions'] = self.recipient_lesions
