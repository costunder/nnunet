"""DEBUG regression checks on the preserved archive's real geometry functions.

Small synthetic masks are isolated unit fixtures, never experiment samples.
The tests exercise the actual archived coordinate discretisation and validators.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import functools
import linecache
from pathlib import Path
import types
import unittest
from unittest import mock
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi

from hiercp_v1x import bounded_scope as scope
from tools import v23_radius_context as policy


ROOT = Path(__file__).resolve().parents[1]


def archived_geometry():
    """Execute only the relevant exact archive functions in private globals."""
    filename = str(ROOT / 'versions/v1/pipeline_v1_source.zip') + ':hiercp/spatial.py'
    with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
        source = archive.read('hiercp/spatial.py').decode().replace('\r\r\n', '\n').replace('\r\n', '\n')
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    names = {'_cfg', '_field', '_unique_rows', 'surface_mask', 'voxel_grid_coordinates',
             'canonical_coordinate_sets', 'validate_canonical_coordinates'}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise AssertionError('Required geometry functions are missing from the archive')
    module = types.ModuleType('radius_context_debug_archived_geometry')
    module.np, module.ndi = np, ndi
    module.EmptyCanonicalNodeError = ValueError
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, filename, 'exec'), vars(module))
    return module


def debug_config():
    # These values preserve the checkpoint's coordinate scales; the masks alone
    # are reduced unit fixtures and do not replace a final execution profile.
    return types.SimpleNamespace(context_inner_radius_mm=2., context_outer_radius_mm=10.,
        context_radius_mm=28., context_shells_mm=(4., 12., 28.),
        boundary_depth_mm=3., context_liver_surface_separation_mm=1.,
        canonical_surface_spacing_mm=2., canonical_interior_spacing_mm=3.,
        canonical_context_spacing_mm=2.5, canonical_liver_spacing_mm=4.,
        canonical_node_limit=250_000, adaptive_roi_margin_mm=10., patch_size=48,
        validate=lambda: None)


def debug_fields(*, shallow=False, spacing=(1., 1., 1.)):
    footprint = np.zeros((31, 31, 31), dtype=bool)
    footprint[14:17, 14:17, 14:17] = True
    organ = np.ones_like(footprint)
    outside = ndi.distance_transform_edt(~footprint, sampling=spacing).astype(np.float32)
    depth = np.full(footprint.shape, 2. if shallow else 8., np.float32)
    depth[0:3] = 1.
    return types.SimpleNamespace(footprint=footprint, organ_mask=organ,
        outside_tumor_mm=outside, organ_depth=depth)


def patched_geometry():
    module = archived_geometry()
    module.validate_canonical_coordinates = scope._patched(
        module.validate_canonical_coordinates, policy._context_validator, 3)
    module.canonical_coordinate_sets = scope._patched(
        module.canonical_coordinate_sets, policy._context_builder, 1)
    return module


@contextmanager
def installed_geometry():
    """Exercise production install over the original bounded-scope guard."""
    spatial = archived_geometry()
    original_validator = spatial.validate_canonical_coordinates
    optional_validator = scope._patched(original_validator, scope._allow_empty_coordinates, 1)
    @functools.wraps(original_validator)
    def bounded_validator(fields, coordinates, config):
        scope._scope_config(config, 10.)
        optional_validator(fields, coordinates, config)
        if len(coordinates['liver_surface']) == 0:
            liver_surface = fields.organ_mask & ~fields.footprint & (fields.organ_depth > 0.) & (
                fields.organ_depth <= config.boundary_depth_mm)
            if np.any(liver_surface):
                raise ValueError('A present real liver surface cannot be silently omitted')
    spatial.validate_canonical_coordinates = bounded_validator
    def rebind(original, replacement):
        for name, value in list(vars(spatial).items()):
            if value is original:
                setattr(spatial, name, replacement)
    runtime = dict(spatial=spatial, scope=dict(margin_mm=10., contract_sha256='DEBUG_OPTIONAL_SURFACE'))
    with mock.patch.object(policy, '_INSTALLED', None), mock.patch.object(scope, '_rebind', side_effect=rebind):
        policy.install(runtime)
        yield spatial


class RadiusContextDebug(unittest.TestCase):
    def test_shallow_and_nearby_actual_voxels_are_included_above_1mm_depth(self):
        geometry, config = patched_geometry(), debug_config()
        for spacing in ((1., 1., 1.), (.7, 1.3, 2.5)):
            fields = debug_fields(shallow=True, spacing=spacing)
            original = {key: value.copy() for key, value in vars(fields).items()}
            points = geometry.canonical_coordinate_sets(fields, config, spacing)
            eligible = (fields.organ_mask & ~fields.footprint
                        & (fields.outside_tumor_mm <= 10.) & (fields.organ_depth > 1.))
            expected = geometry.voxel_grid_coordinates(
                np.argwhere(eligible), np.asarray(spacing, np.float32), 2.5)
            np.testing.assert_array_equal(points['context'], expected)
            distance = fields.outside_tumor_mm[tuple(points['context'].T)]
            self.assertTrue(np.any(distance < 2.))
            self.assertTrue(np.all(distance <= 10.))
            self.assertTrue(np.all(fields.organ_depth[tuple(points['context'].T)] > 1.))
            self.assertTrue(np.all(fields.organ_depth[tuple(points['context'].T)] < 4.))
            for key, value in original.items():
                np.testing.assert_array_equal(getattr(fields, key), value)

    def test_builder_excludes_exact_1mm_and_includes_next_float32_depth(self):
        geometry, config, fields = patched_geometry(), debug_config(), debug_fields(shallow=True)
        fields.organ_depth.fill(1.)
        exact_boundary = (10, 15, 15)
        just_inside = (14, 10, 15)
        ordinary_inside = (14, 15, 10)
        fields.organ_depth[just_inside] = np.nextafter(np.float32(1.), np.float32(np.inf))
        fields.organ_depth[ordinary_inside] = 2.
        points = geometry.canonical_coordinate_sets(fields, config, (1., 1., 1.))
        rows = {tuple(row) for row in points['context']}
        self.assertNotIn(exact_boundary, rows)
        self.assertEqual(rows, {just_inside, ordinary_inside})
        geometry.validate_canonical_coordinates(fields, points, config)
        self.assertEqual(config.context_inner_radius_mm, 2.)

    def test_all_noncontext_node_tables_and_discretisation_remain_exact(self):
        original, updated = archived_geometry(), patched_geometry()
        fields, config, spacing = debug_fields(), debug_config(), (.8, .9, 1.5)
        before = original.canonical_coordinate_sets(fields, config, spacing)
        after = updated.canonical_coordinate_sets(fields, config, spacing)
        self.assertEqual(set(before), set(after))
        for role in ('surface', 'interior', 'liver_surface'):
            np.testing.assert_array_equal(before[role], after[role])
        self.assertGreater(len(after['context']), len(before['context']))

    def test_cross_role_overlap_is_allowed_but_duplicate_context_is_rejected(self):
        geometry, fields, config = patched_geometry(), debug_fields(shallow=True), debug_config()
        points = geometry.canonical_coordinate_sets(fields, config, (1., 1., 1.))
        rows = {tuple(row) for row in points['context']}
        overlap = next(row for row in points['liver_surface'] if tuple(row) in rows)
        geometry.validate_canonical_coordinates(fields, points, config)
        invalid = dict(points, context=np.concatenate([points['context'], overlap[None]], axis=0))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            geometry.validate_canonical_coordinates(fields, invalid, config)

    def test_physical_upper_limit_nonorgan_and_tumor_membership_remain_checked(self):
        geometry, config = patched_geometry(), debug_config()
        for violation in ('outside', 'negative_distance', 'nonorgan', 'footprint',
                          'exact_1mm_depth', 'below_1mm_depth', 'out_of_bounds', 'empty'):
            with self.subTest(violation=violation):
                fields = debug_fields(shallow=True)
                points = geometry.canonical_coordinate_sets(fields, config, (1., 1., 1.))
                invalid = {key: value.copy() for key, value in points.items()}
                coordinate = invalid['context'][0]
                if violation == 'outside':
                    fields.outside_tumor_mm[tuple(coordinate)] = 10.01
                elif violation == 'negative_distance':
                    fields.outside_tumor_mm[tuple(coordinate)] = -1.
                elif violation == 'nonorgan':
                    fields.organ_mask[tuple(coordinate)] = False
                elif violation == 'footprint':
                    invalid['context'][0] = invalid['surface'][0]
                elif violation == 'exact_1mm_depth':
                    fields.organ_depth[tuple(coordinate)] = 1.
                elif violation == 'below_1mm_depth':
                    fields.organ_depth[tuple(coordinate)] = np.nextafter(np.float32(1.), np.float32(0.))
                elif violation == 'out_of_bounds':
                    invalid['context'][0] = (-1, 0, 0)
                else:
                    invalid['context'] = np.empty((0, 3), np.int64)
                with self.assertRaises(ValueError):
                    geometry.validate_canonical_coordinates(fields, invalid, config)

    def test_builder_excludes_outside_nonorgan_and_exact_footprint_voxels(self):
        geometry, config, fields = patched_geometry(), debug_config(), debug_fields(shallow=True)
        fields.organ_mask[13, 15, 15] = False
        points = geometry.canonical_coordinate_sets(fields, config, (1., 1., 1.))
        indexes = tuple(points['context'].T)
        self.assertTrue(np.all(fields.organ_mask[indexes]))
        self.assertFalse(np.any(fields.footprint[indexes]))
        self.assertFalse(np.any(fields.outside_tumor_mm[indexes] > 10.))
        self.assertTrue(np.all(fields.organ_depth[indexes] > 1.))

    def test_malformed_shapes_and_other_roles_are_still_rejected(self):
        geometry, config, fields = patched_geometry(), debug_config(), debug_fields(shallow=True)
        points = geometry.canonical_coordinate_sets(fields, config, (1., 1., 1.))
        with self.assertRaisesRegex(ValueError, 'shape'):
            bad_fields = types.SimpleNamespace(**dict(vars(fields), organ_depth=np.zeros((2, 2), np.float32)))
            geometry.canonical_coordinate_sets(bad_fields, config, (1., 1., 1.))
        for role, wrong in [('context', np.zeros((1, 2), np.int64)),
                            ('surface', np.array([[0, 0, 0]], np.int64)),
                            ('liver_surface', np.array([[15, 15, 15]], np.int64))]:
            with self.subTest(role=role), self.assertRaises(ValueError):
                geometry.validate_canonical_coordinates(fields, dict(points, **{role: wrong}), config)

    def test_install_rebinds_shared_source_and_target_builder_and_scopes_validation(self):
        spatial = archived_geometry()
        original_builder, original_validator = spatial.canonical_coordinate_sets, spatial.validate_canonical_coordinates
        # The bounded scope wraps the archived validator before this adapter.
        # Keep that real call shape and unwrap chain in the isolated fixture.
        @functools.wraps(original_validator)
        def bounded_validator(fields, coordinates, config):
            return original_validator(fields, coordinates, config)
        spatial.validate_canonical_coordinates = bounded_validator
        local = types.SimpleNamespace(canonical_coordinate_sets=original_builder)
        bindings = []
        def rebind(original, replacement):
            bindings.append((original, replacement))
            for module in (spatial, local):
                for name, value in list(vars(module).items()):
                    if value is original:
                        setattr(module, name, replacement)
        runtime = dict(spatial=spatial, scope=dict(margin_mm=10., contract_sha256='DEBUG_SCOPE'))
        with mock.patch.object(policy, '_INSTALLED', None), mock.patch.object(scope, '_rebind', side_effect=rebind):
            report = policy.install(runtime)
            self.assertEqual(len(bindings), 2)
            self.assertIs(local.canonical_coordinate_sets, spatial.canonical_coordinate_sets)
            self.assertEqual(report, policy.identity())
            self.assertEqual(policy.install(runtime), report)
            fields, config = debug_fields(shallow=True), debug_config()
            points = local.canonical_coordinate_sets(fields, config, (1., 1., 1.))
            spatial.validate_canonical_coordinates(fields, points, config)
            wrong_config = debug_config()
            wrong_config.context_outer_radius_mm = 11.
            with self.assertRaisesRegex(ValueError, 'scope'):
                local.canonical_coordinate_sets(fields, wrong_config, (1., 1., 1.))
            with self.assertRaises(RuntimeError):
                policy.install(dict(spatial=spatial, scope=dict(margin_mm=10., contract_sha256='OTHER_SCOPE')))
        with ZipFile(ROOT / 'versions/v1/pipeline_v1_source.zip') as archive:
            source = archive.read('hiercp/local.py').decode().replace('\r\r\n', '\n').replace('\r\n', '\n')
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        for name in ('prepare_local_source', '_prepare_local_target'):
            calls = [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'canonical_coordinate_sets']
            self.assertEqual(len(calls), 1)

    def test_archive_ast_drift_fails_explicitly(self):
        geometry = archived_geometry()
        node = scope._function_ast(geometry.canonical_coordinate_sets)
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == 'context_mask' for target in child.targets):
                child.value = ast.parse('organ & ~footprint', mode='eval').body
        with self.assertRaisesRegex(ValueError, 'expression differs'):
            policy._context_builder(node)

    def test_installed_policy_allows_empty_liver_surface_only_if_actual_surface_absent(self):
        with installed_geometry() as spatial:
            fields, config = debug_fields(), debug_config()
            fields.organ_depth.fill(8.)
            points = spatial.canonical_coordinate_sets(fields, config, (1., 1., 1.))
            self.assertEqual(points['liver_surface'].shape, (0, 3))
            self.assertGreater(len(points['context']), 0)
            spatial.validate_canonical_coordinates(fields, points, config)

    def test_installed_policy_rejects_omitted_present_liver_surface(self):
        with installed_geometry() as spatial:
            fields, config = debug_fields(shallow=True), debug_config()
            points = spatial.canonical_coordinate_sets(fields, config, (1., 1., 1.))
            self.assertGreater(len(points['liver_surface']), 0)
            omitted = dict(points, liver_surface=np.empty((0, 3), np.int64))
            with self.assertRaisesRegex(ValueError, 'present real liver surface'):
                spatial.validate_canonical_coordinates(fields, omitted, config)


if __name__ == '__main__':
    unittest.main()
