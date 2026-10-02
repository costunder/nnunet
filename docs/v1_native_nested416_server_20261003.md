# Native v1.0 / strict-nested416 서버 순위 대조

사용자가 2026-10-03 승인한 비교를 고정한다. sampler와 모델을 더 바꾸지 않고, native v1.0과 그 sampled view 내부의 strict-nested416을 각각 seed42 / 40 epochs로 학습한다. 이번 추가는 양쪽에 동일한 epoch 계측과 실행 로그를 붙이는 것이다. 기존 r5 ZIP을 다시 만들거나 그 GPU evidence를 새 실행 결과로 바꾸지 않는다.

## 고정한 계약

| 항목 | 두 arm의 공통 값 |
| --- | --- |
| 모델 | 보존 원본 v1.0, 10,434,532 parameters |
| CNN / GAT | 원본 CNN base12 / feature32, hidden128, heads4, local3 / patient2 / population2 |
| 정답 | source tumor의 실제 original anchor; 원본 corruption/curriculum 비교 |
| 학습 | seed42, 40 epochs, 원본 loss / optimizer / scheduler / AMP |
| 후보 | pool128, 원본 curriculum 8개/sample, 두 view |
| 분할 | 84 train / 21 validation / outer26 제외 |
| 준비 | native의 canonical cache / prototype bank 한 벌 공유 |
| batch / worker | native의 실제 preflight 측정값을 공통 execution lock으로 고정 |
| gradient accumulation | 원본 1; effective batch = physical batch, 단일 선택 GPU |
| masks | 원본 full source footprint / 원본 eligibility 유지 |

유일한 연구 변인은 local sampled graph이다. Native는 원본 sampled view이고 nested는 그 view의 strict subset과 induced edges다. Nested role seed budget은 tumor surface/interior, source context/liver surface, target context/liver surface 순서로 `64 / 32 / 96 / 64 / 96 / 64`이다. 416은 seed budget 합이며 최종 node cap이나 최적값을 뜻하지 않는다. 104, relay, profile 자동 조정, v2.2 P/U 재정의는 적용하지 않는다.

## 실행과 기록

`tools/run_v1x_experiment.py init --local-sampling native --record-epochs`로 새 native experiment를 만든다. `run --stage v1.0 --target prepare`가 원본 준비를 실행하며, 이어지는 `--target train --gpu <번호>`가 원본 auto preflight와 40-epoch 학습을 실행한다. GPU 번호는 실행 환경의 `nvidia-smi -L`에 표시된 번호 하나로 지정한다. 선택은 실시간 inventory로 확인하며 UUID를 실행 명령에 고정하지 않는다. 단일 GPU만 표시하는 컨테이너에서는 그 환경에 표시된 번호를 사용한다.

Native preflight가 출판되면 첫 optimization epoch 전에 `native/shared/execution_lock.json`을 만든다. Native 초기 config의 batch/worker는 그대로 auto이며, nested `resolved_config.json`만 native의 측정된 정수값을 사용한다. Nested는 `--reference-experiment <native>`와 같은 `--record-epochs`를 사용한다. 준비를 다시 만들지 않으며 기존 v2.2 paired cache를 대신 채택하지 않는다.

두 checkpoint 위치는 독립적이다.

- Native: `<comparison>/native/results/v1.0/checkpoint_best.pt` 및 `checkpoint_best.last.pt`.
- Nested: `<comparison>/nested416/results/v1.0/checkpoint_best.pt` 및 `checkpoint_best.last.pt`.

각 results/v1.0에는 invocation별 `epoch_telemetry_*.jsonl`, `invocation_*.stdout.log`, `invocation_*.stderr.log`가 별도로 남는다. 화면에는 batch 진행률과 원본 epoch 품질 지표가 표시된다. 재호출의 로그는 새 파일을 만들며 이전 로그를 덮어쓰지 않는다.

| 요구한 기록 | 실제 필드 |
| --- | --- |
| total / ranking / pair loss | `train.loss / ranking / pair` 및 원본 `ce / ordinal / mined / consistency` |
| train MRR / positive-best-other margin | `train.mrr / margin`, `positive / hardest_negative` |
| validation MRR / top1 / margin | `validation.mrr / acc / margin` (`acc`가 원본 top1) |
| epoch wall | `telemetry.epoch_wall_seconds` |
| loader 대기 | 각 pass의 `loader_wait_seconds` |
| 실제 view sampling / collate | `sampling_worker_seconds_sum`, min/max/count, `collate_worker_seconds_sum` |
| GPU optimization / validation compute | CUDA events의 `gpu_optimization_seconds / gpu_validation_compute_seconds` |
| validation wall | 원본 `validation.elapsed_seconds` |
| checkpoint 저장 | `telemetry.checkpoint_save_seconds`, 저장별 경로·시간 |
| peak VRAM / RAM / CPU | 원본 pass resource metrics 및 `telemetry.epoch_peak_vram_allocated_bytes` |

Worker sampling 시간 합은 병렬 worker의 작업량이다. Loader 대기·GPU 작업과 겹치므로 epoch wall에 단순히 더하지 않는다. CUDA events는 현재 stream의 forward/loss/backward/optimizer를 재며 metrics 및 별도 H2D prefetch stream을 포함하지 않는다. 새 batch별 synchronize나 loss `.item()`은 추가하지 않는다. Loader wrapper는 다음 fetch 전에 이전 batch 참조를 해제한다.

## 중단과 후속 평가

Foreground 작업이므로 Ctrl+C는 원본 작업을 중단한다. 원본 v1의 restart point는 완결 train+validation epoch 뒤의 last checkpoint다. 미완결 epoch의 batch 진행을 저장했다고 주장하지 않는다. **첫 epoch 이전에 중단하여 preflight만 있고 last checkpoint가 없는 경우 원본의 자동 재시작 제한이 남아 있다.** 이 상태를 숨기거나 preflight를 삭제/재작성하지 않는다. 완료된 native를 다시 학습하는 대신 그 결과를 확인하고 nested 단계부터 진행한다.

두 40-epoch 결과를 완료 검증한 후 `tools/evaluate_v1_sampling_pair.py`만 실행한다. 동일 전체21 validation case의 sample identity와 분모를 확인하고 case별 MRR/top1/margin을 집계한다. Case mean 차이의 paired percentile bootstrap은 실행 명령에서 **20,000 resamples / 95% confidence / seed42**를 명시한다. 이는 21-case CI의 Monte Carlo 오차를 줄이기 위한 통계 설정이며 학습 반복 수 변경이나 품질 허용 오차가 아니다. 평가 GPU/RSS 예산도 명시한다.

Best를 선택한 validation을 재평가하므로 CI는 그 선택에 조건부인 결과다. 독립 test 정확도나 CP 효용으로 해석하지 않는다. 결과의 차이와 CI를 보고하며 자동 `quality PASS`나 최적 graph 크기 판정을 만들지 않는다. 전체128 production CP, Basic CP80/nnU-Net, v2.2 결합은 이번 실행에 없다.

## 검증의 범위

이번 metadata/source/runtime 검사와 실제 CT CUDA 계측 검사의 기록은 `validation/v1x_progressive_20261003/server_epoch_recording_release.json`에 보존한다. 87개 metadata UNIT 및 14개 Python AST 검사가 통과했고, 실제 CT/GPU에서는 native와416 각각 원본·원본 반복·계측의 한 batch씩 실행했다. 두 경로의 1,085개 trainable parameter tensor에 모두 gradient가 연결됐다. 원시 수치 JSON은 같은 validation 디렉터리의 `epoch_recording/`에 포함한다. 소규모 신경 실행은 DEBUG 전용이고 full model / 8 candidates / two views를 사용한다. 최종 학습 profile에는 debug dataset이나 batch를 반영하지 않는다. Bash launcher는 LF·CLI 인수를 검토했으며 로컬 Windows에 Bash가 없어 현지 실행은 하지 않았다.

원본 v1은 cuDNN deterministic 설정을 쓰지만 모든 CUDA 연산의 bit 재현성을 보장하지 않는다. 실제 비교에서 AMP 수치의 작은 변동을 확인했고, 전역 strict deterministic DEBUG는 원본 `grid_sampler_3d_backward_cuda`의 미지원으로 실패했다. 실패를 production 설정 변경으로 우회하지 않는다. 반복 원본과 계측 경로의 수치·전체 gradient 연결·AdamW 업데이트 방향·RNG 상태를 기록하는 mechanical 검사를 사용한다. 수치 허용 범위는 AMP dtype epsilon에서 정하고 ranking 품질 허용 오차로 전용하지 않는다. Bitwise numerical parity와 전체 장기 학습 품질은 증명했다고 표시하지 않는다.

기존 r5 전달본 SHA256은 `ef5814d011889dc2ad14c453180c33f63eb0df7ae4d8e61fb29b1a75b5af418e`이며 그대로 보존한다. 추가 계측 helper는 새로운 server invocation source contract에 결속되므로 r5의 옛 workspace hash와 새 helper hash를 같다고 주장하지 않는다. 보존 원본 v1 ZIP, 모델 원본 및 strict sampler bytes는 유지한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인된 416 대조만 사용한다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. Native 측정 lock을 공유한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 로컬 DEBUG 자원과 서버 preflight를 구분한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 모델 축소를 추가하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 CUDA DEBUG 범위로 검증한다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 장기 실행은 사용자가 서버에서 시작한다.
