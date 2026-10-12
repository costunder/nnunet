"""Read the authorized B-only transition through the unchanged v2.5 UI2.

Historical monitors, checkpoints and public viewers are never written. GNN
partial completion stays distinct from full native250/BEST26 completion.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

import psutil

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from tools import watch_v25_pipeline as original
from tools import watch_v25_all_arm_recovery as previous

ROUTE_FORMAT = 'v25_B_user_transition_viewer_route_v1'
EARLY_FORMAT = 'v25_exact_B_authorized_early_downstream_v1'
PIN_FORMAT = 'v25_user_authorized_B_partial_own_full128_BEST_v1'
RELEASE_FORMAT = 'v25_user_B_only_checkpoint_stop_v1'
PAUSED_ARMS = ('A', 'C', 'D')


def observe_predecessor(declared, path):
    actual, problem = original.observe_actor(declared, commands=[declared.get('command')], path=path)
    if actual is not None:
        return actual, problem
    # The common observer treats zombies as absent. Check the exact birth/UID
    # before accepting that terminal observation, including surviving workers.
    try:
        process = psutil.Process(declared['pid'])
        uid = process.uids().real if hasattr(process, 'uids') else None
        if abs(process.create_time()-declared['create_time']) > .001 or uid != declared['UID']:
            raise ValueError('Paused predecessor PID was reused or its exact birth/UID differs')
        if process.status() != psutil.STATUS_ZOMBIE:
            raise ValueError('Paused predecessor changed during exact terminal observation; retry')
        active = []
        for child in process.children(recursive=True):
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    child_uid = child.uids().real if hasattr(child, 'uids') else None
                    if child_uid != uid:
                        raise ValueError('Paused predecessor has a descendant with a different UID')
                    active.append(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if active:
            return dict(declared, process_status='zombie; owned workers closing', RSS_bytes=sum(active)), None
        return None, problem
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None, problem


class Transition:
    def __init__(self, group, old_seal, old_sha, authorization, authorization_sha, route):
        self.group = original.input_path(group)
        self.old_seal = original.input_path(old_seal)
        self.old_sha = old_sha
        self.monitors = previous.monitor_bindings(self.group, self.old_seal, old_sha)
        self.authorization = original.input_path(authorization)
        self.namespace = self.authorization.parent
        self.route = original.input_path(route)
        if (self.authorization.name != 'request.json' or self.namespace.parent != self.group
                or not self.namespace.name.startswith('B_only_user_transition_')
                or self.route != self.namespace/'B_viewer_route.json'):
            raise ValueError('Exact separate B-only authorization/route namespace required')
        self.cache = {}
        self.auth = self.proof(dict(path=str(self.authorization), sha256=authorization_sha))
        if (self.auth.get('group_root') != str(self.group)
                or self.auth.get('original_seal_path') != str(self.old_seal)
                or self.auth.get('GNN_only_early_stop_authorized') is not True
                or self.auth.get('CP_sources') != 642 or self.auth.get('CP_candidates_per_source') != 128
                or self.auth.get('native_epochs') != 250 or self.auth.get('evaluation_cases') != 26
                or self.auth.get('evaluation_checkpoint') != 'checkpoint_best.pth'
                or type(self.auth.get('B_finish_epoch')) is not int
                or not 1 <= self.auth['B_finish_epoch'] < 40
                or not isinstance(self.auth.get('user_instruction'), str)
                or not self.auth['user_instruction'].strip()
                or set(self.auth.get('actors', {})) != set(previous.GPUS)):
            raise ValueError('Explicit original B-only/partialGNN/full642/native250/BEST26 request required')
        self.authorization_proof = dict(path=str(self.authorization), sha256=authorization_sha)

    def proof(self, ref):
        if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
                or not isinstance(ref['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', ref['sha256'])):
            raise ValueError('Exact metadata path/SHA256 proof required')
        path = original.input_path(ref['path'])
        if not path.is_relative_to(self.group):
            raise ValueError('B-only metadata proof escaped the original group')
        self.witness(path, ref['sha256'])
        return original.read(path, required=True)

    def witness(self, path, checksum):
        path = original.input_path(path)
        if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Explicit immutable byte witness required')
        info = path.stat()
        key = (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if key not in self.cache:
            self.cache[key] = original.sha(path)
        if self.cache[key] != checksum:
            raise ValueError('B-only immutable witness changed: '+str(path))

    def early(self):
        if not self.route.exists():
            return None
        route = original.read(self.route, required=True)
        if (route.get('format') != ROUTE_FORMAT
                or route.get('authorization_request') != self.authorization_proof):
            raise ValueError('Published B-only viewer pointer differs from the explicit authorization')
        seal = self.proof(route.get('early_seal'))
        if (seal.get('format') != EARLY_FORMAT or seal.get('selected_arms') != ['B']
                or seal.get('group_root') != str(self.group)
                or seal.get('authorization_request') != self.authorization_proof
                or seal.get('previous_all_seal') != dict(path=str(self.old_seal), sha256=self.old_sha)
                or seal.get('GNN_early_stop_authorized') is not True
                or seal.get('scientific_configuration_unchanged') is not True
                or seal.get('processes_signaled_by_this_adapter') is not False):
            raise ValueError('Exact own B authorized downstream seal required')
        slot = seal.get('slot') or {}
        monitor = original.input_path(slot.get('monitor_path', ''))
        if (slot.get('arm') != 'B' or slot.get('gpu') != 5 or slot.get('CPU_affinity') != list(range(42, 48))
                or monitor != self.namespace/'execution/monitor/status.json'):
            raise ValueError('Fresh separate own B early downstream monitor required')
        science = self.proof(seal.get('scientific_request'))
        cp_root = original.input_path(science.get('cp_output', ''))
        if (science.get('source_gnn_output') != str(self.group/'B')
                or not cp_root.is_relative_to(self.namespace) or cp_root.name != 'cp_bank'):
            raise ValueError('Fresh authorized B CP namespace and original own GNN output required')
        release = self.proof(seal.get('release_receipt'))
        self.validate_release(release)
        return dict(seal=seal, science=science, monitor=monitor, cp_root=cp_root, release=release)

    def validate_release(self, release):
        if (release.get('format') != RELEASE_FORMAT or release.get('status') != 'PAUSED_ACD_B_READY_FOR_OWN_BEST'
                or release.get('selected_downstream_arms') != ['B']
                or release.get('paused_arms') != list(PAUSED_ARMS)
                or release.get('user_instruction') != self.auth['user_instruction']
                or any(release.get(key) is not True for key in ('old_controller_released',
                    'registered_stage_actors_released', 'all_GNN_children_stopped_cooperatively'))
                or set(release.get('states', {})) != set(previous.GPUS)):
            raise ValueError('Actual exact cooperative ACD/B predecessor release receipt required')
        for arm, state in release['states'].items():
            if (state.get('status') != 'PAUSED' or state.get('latest_path') != str(self.group/arm/'training/checkpoint_latest.pt')
                    or type(state.get('completed_epochs')) is not int or not 0 <= state['completed_epochs'] < 40
                    or type(state.get('updates')) is not int or state['updates'] < 0
                    or type(state.get('epoch')) is not int or not 1 <= state['epoch'] <= 40):
                raise ValueError('Truthful original paused checkpoint state required')
        if release['states']['B']['completed_epochs'] < self.auth['B_finish_epoch']:
            raise ValueError('B did not finish the explicitly authorized current epoch')

    def pause(self, arm):
        self.witness(self.old_seal, self.old_sha)
        self.witness(self.authorization, self.authorization_proof['sha256'])
        rows, _ = original.original.JsonlCursor().read(self.group/arm/'training/invocations.jsonl')
        invocation = rows[-1] if rows else None
        declared = self.auth['actors'][arm]
        actual, problem = observe_predecessor(declared, self.authorization)
        if invocation is None or invocation.get('status') != 'PAUSED':
            return dict(status='PAUSE_REQUESTED_PENDING' if arm in PAUSED_ARMS else None,
                reason='사용자 중단 요청 확인; 실제 PAUSED checkpoint 기록 대기' if arm in PAUSED_ARMS else None,
                actor=actual, invocation=None, process_released=actual is None)
        path = original.input_path(invocation.get('checkpoint', ''))
        if (invocation.get('format') != original.GNN_FORMAT or invocation.get('full_training') is not False
                or invocation.get('debug') is not False or invocation.get('actual_CUDA') is not True
                or invocation.get('connected_parameter_tensors') != 537 or invocation.get('expected_parameter_tensors') != 537
                or type(invocation.get('completed_epochs')) is not int or not 0 <= invocation['completed_epochs'] < 40
                or type(invocation.get('optimizer_updates')) is not int or invocation['optimizer_updates'] < 0
                or path != self.group/arm/'training/checkpoint_latest.pt' or not path.is_file()):
            raise ValueError('Actual unchanged saved PAUSED full-model checkpoint invocation required')
        reason = '사용자 요청으로 GNN 중단; 저장된 checkpoint 보존'
        if actual is not None:
            status = 'PAUSED_USER_REQUEST_CLOSING'
            reason += '; 원 학습 프로세스 종료 중, 자원 해제 미확인'
        elif arm == 'B':
            status = 'WAITING_FOR_BEST_HANDOFF'
            reason += '; 자기 full128 BEST 후속 CP 연결 대기'
        else:
            status = 'PAUSED_USER_REQUEST'
            reason += '; 원 학습 actor 종료 확인'
        return dict(status=status, reason=reason, actor=actual, invocation=invocation,
            process_released=actual is None, actor_problem=problem)


class Viewer(original.Viewer):
    def __init__(self, group, arm, transition):
        super().__init__(group, arm)
        self.transition = transition
        self.gnn_only = False

    def scoring(self, output, arm, commands, *, log=None):
        return None if self.gnn_only else super().scoring(output, arm, commands, log=log)

    def native(self, output, arm, commands):
        return None if self.gnn_only else super().native(output, arm, commands)

    def _snapshot(self):
        self.gnn_only = True
        try:
            value = super()._snapshot()
        finally:
            self.gnn_only = False
        arm = value['arm']
        early = self.transition.early()
        paused = self.transition.pause(arm)
        if early and paused['invocation']:
            saved = early['release']['states'][arm]
            if (paused['invocation']['completed_epochs'] != saved['completed_epochs']
                    or paused['invocation']['optimizer_updates'] != saved['updates']):
                raise ValueError('Current paused invocation differs from the sealed released checkpoint counters')
        if paused['status']:
            value.update(status=paused['status'], native_pending_reason=paused['reason'],
                original_monitor_frozen=True, GNN_process_released=paused['process_released'])
            gnn = value['gnn']
            gnn.update(status=paused['status'], full_training_completed=False,
                process_status=(paused['actor'] or {}).get('process_status'))
            if paused['invocation']:
                invocation = paused['invocation']
                gnn.update(completed_epochs=invocation['completed_epochs'],
                    optimizer_updates=invocation['optimizer_updates'], active_U=invocation.get('active_u'),
                    stage='paused', stage_label='사용자 요청으로 GNN 중단', progress=None)
            gnn.pop('error', None)
        if arm != 'B' or early is None:
            return value
        state = original.read(early['monitor'], required=True)
        request = state.get('request') or {}
        declarations = request.get('stages') or []
        if (state.get('arm') != 'B' or state.get('GPU') != 5
                or request.get('arm') != 'B' or request.get('GPU') != 5
                or request.get('production_output') != str(self.group/'B')
                or request.get('GNN_full_training') is not False
                or request.get('RAM_GiB') != (96 if state.get('stage') == 'native' else 64)
                or [item.get('name') for item in declarations] != ['scoring', 'native']):
            raise ValueError('Truthful own B CP/native stage request required')
        commands = [item.get('command') for item in declarations]
        if any(not isinstance(command, list) or not command
                or any(not isinstance(argument, str) or not argument for argument in command) for command in commands):
            raise ValueError('Both B-only downstream stage commands require complete declared argv')
        actor, problem = original.observe_actor(original.declared_actor(state), commands=commands, path=early['monitor'])
        status = state.get('status')
        if status == 'RUNNING' and problem:
            status = problem
        released = early['release']['states']['B']
        if (request.get('GNN_completed_epochs') != released['completed_epochs']
                or request.get('GNN_actual_updates') != released['updates']):
            raise ValueError('Actual B stopped epoch/update counters differ from release proof')
        value['gnn'].update(completed_epochs=released['completed_epochs'], optimizer_updates=released['updates'],
            epoch=released['epoch'], epochs=40, full_training_completed=False,
            status='PAUSED_USER_REQUEST', stage='paused', stage_label='사용자 승인 GNN 조기 종료', progress=None)
        value.update(status=status, native_pending_reason=state.get('error'), GNN_process_released=True,
            original_monitor_frozen=True, authorized_B_downstream=True)
        cp = self.early_scoring(early, commands, state)
        if cp is not None:
            value['scoring'] = cp
        native = super().native(self.group/'B', 'B', commands)
        if native is not None:
            value['native'] = native
        if state.get('stage') in ('native', 'complete'):
            value.update(active_stage='native', status=(native or {}).get('status', status),
                full_native_training=(native or {}).get('full_native_training') is True,
                full_native_evaluation=(native or {}).get('full_native_evaluation') is True)
        else:
            value.update(active_stage='scoring', status=(cp or {}).get('status', status))
            if state.get('status') == 'FAILED':
                value['status'] = 'FAILED'
        if state.get('error'):
            value['gnn']['error'] = state['error']
        return value

    def early_scoring(self, early, commands, state):
        root = early['cp_root']; path = root/'progress.json'
        progress = original.read(path)
        if progress is None:
            return None
        request = original.read(root/'request.json', required=True)
        pin = request.get('best_provenance') or {}
        completed = early['release']['states']['B']['completed_epochs']
        updates = early['release']['states']['B']['updates']
        best_path = self.group/'B/training/checkpoint_best.pt'
        if (progress.get('format') != original.CP_PROGRESS or progress.get('arm') != 'B'
                or request.get('format') != original.CP_REQUEST
                or progress.get('request_sha256') != original.sha(root/'request.json')
                or progress.get('total_sources') != 642 or progress.get('total_positions') != 82176
                or type(progress.get('completed_sources')) is not int or not 0 <= progress['completed_sources'] <= 642
                or progress.get('completed_positions') != 128*progress['completed_sources']
                or pin.get('format') != PIN_FORMAT or pin.get('arm') != 'B' or pin.get('physical_GPU') != 5
                or pin.get('full_training') is not False or pin.get('debug') is not False
                or pin.get('completed_epochs') != completed or pin.get('optimizer_updates') != updates
                or pin.get('original_planned_epochs') != 40 or pin.get('original_planned_optimizer_updates') != 680
                or pin.get('source_completion_kind') != 'USER_AUTHORIZED_EPOCH_BOUNDARY_EARLY_STOP'
                or pin.get('selected_by') != 'fixed_full128_validation_only'
                or pin.get('checkpoint') != str(best_path)
                or pin.get('checkpoint_sha256') != early['seal']['preserved_files'].get(str(best_path))
                or pin.get('original_patient_and_full128_history_verified') is not True
                or pin.get('source_checkpoint_written') is not False
                or pin.get('source_model_optimizer_RNG_rewritten') is not False):
            raise ValueError('Truthful authorized partial B BEST/full642 durable scoring binding differs')
        self.transition.witness(best_path, pin['checkpoint_sha256'])
        source_authorization = pin.get('authorization') or {}
        source_ref = dict(path=source_authorization.get('path'), sha256=source_authorization.get('sha256'))
        declared_ref = early['science'].get('authorization_request')
        if early['science'].get('format') == 'v25_user_authorized_B_epoch_boundary_stop_v1':
            declared_ref = early['seal']['scientific_request']
        if source_ref != declared_ref:
            raise ValueError('Own partial BEST scientific authorization differs from the sealed request')
        auth = self.transition.proof(source_ref)
        if (auth.get('format') != 'v25_user_authorized_B_epoch_boundary_stop_v1'
                or auth.get('explicit_user_authorization') is not True or auth.get('arm') != 'B'
                or auth.get('physical_GPU') != 5 or auth.get('target_completed_epochs') != completed
                or auth.get('native_epochs') != 250 or auth.get('native_evaluation_cases') != 26
                or auth.get('native_evaluation_checkpoint') != 'checkpoint_best.pth'):
            raise ValueError('Partial BEST requires exact own authorized completed epoch/native250/BEST26')
        actor, problem = original.observe_actor(original.declared_actor(progress), commands=commands, path=path)
        status = progress.get('status'); complete = False
        overlay = original.read(root/'score_overlay.json')
        if status == 'COMPLETE':
            if (not overlay or overlay.get('format') != original.CP_OVERLAY or overlay.get('complete') is not True
                    or overlay.get('complete_sources') != 642 or overlay.get('complete_positions') != 82176
                    or overlay.get('debug') is not False or overlay.get('actual_CUDA') is not True
                    or overlay.get('full_CP_scoring_executed') is not True or overlay.get('best_provenance') != pin
                    or overlay.get('durable_score_request_sha256') != progress['request_sha256']
                    or overlay.get('content_sha256') != original.canonical({k:v for k,v in overlay.items() if k != 'content_sha256'})):
                status = 'SCORING_COMPLETION_UNVERIFIED'
            else:
                status = 'SCORING_COMPLETE'; complete = True
        elif status == 'RUNNING' and problem:
            status = problem
        elif status == 'FAILED':
            status = 'SCORING_FAILED'
        log = original.input_path(state['stage_log']) if state.get('stage_log') else None
        if log and not log.is_relative_to(early['monitor'].parent):
            raise ValueError('B scoring console escaped its fresh monitor namespace')
        return dict(status=status, complete=complete, source_done=progress['completed_sources'], source_total=642,
            CP_done=progress['completed_positions'], CP_total=82176, own_BEST_pin_verified=True,
            GNN_full_training=False, GNN_completed_epochs=completed, GNN_optimizer_updates=updates,
            last_batch_seconds=progress.get('last_batch_seconds'),
            seconds_since_console_write=max(0, time.time()-log.stat().st_mtime) if log and log.is_file() else None,
            RSS_GiB=actor['RSS_bytes']/2**30 if actor else None, RAM_limit_GiB=64,
            process_status=(actor or {}).get('process_status'), detail=progress.get('error') or problem)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--recovery-seal', type=Path, required=True)
    parser.add_argument('--recovery-seal-sha256', required=True)
    parser.add_argument('--authorization-request', type=Path, required=True)
    parser.add_argument('--authorization-request-sha256', required=True)
    parser.add_argument('--B-route', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(previous.GPUS), required=True)
    args, remaining = parser.parse_known_args(argv)
    context = Transition(args.group, args.recovery_seal, args.recovery_seal_sha256,
        args.authorization_request, args.authorization_request_sha256, args.B_route)
    normal_read, normal_viewer = original.read, original.Viewer
    routing = {args.group/arm/'monitor/status.json':path for arm,path in context.monitors.items()}
    routing.update({args.group/arm/'monitor/request.json':path.parent/'request.json'
        for arm,path in context.monitors.items()})

    def read(path, *, required=False):
        return normal_read(routing.get(Path(path), Path(path)), required=required)

    original.read = read
    original.Viewer = lambda group, arm=None: Viewer(group, arm, context)
    try:
        return original.main(['--group', str(args.group), '--arm', args.arm, *remaining])
    finally:
        original.read, original.Viewer = normal_read, normal_viewer


if __name__ == '__main__':
    main()
