# v2.2 재계산 비용 비교 — DEBUG 전용

## 후속: 전체 L0 해제 실패와 선택적 후보

사용자가 제공한 서버 결과에서 step114, full support11,279, batch32의 `no_outer_l0`가 첫 후보 batch에서 OOM이었다. 해당 process9.39GiB, PyTorch allocated8.97GiB, free88.06MiB에서162MiB allocation 실패다. 원본 checkpoint hash는 `057d43bf2e2d024a80841b2bb5d86a3fff2ee441f68a400d97915d9a8e50025a`로 보존됐다. 화면의5.72GiB는 baseline 완료 batch 값이지 실패 후보의 peak가 아니다. 이 정책은 production에 적용하지 않았다. 사용자는 이후 `work/v22_gnn40_resume114_20260928`에서 step169까지 기존 학습을 이어간 출력을 제공했다.

기본 비교 후보를 `no_outer_l0_block2`, `no_outer_l0_block1`, `no_outer_l0_block0`으로 변경했다. 각각 세 블록 중 하나만 outer checkpoint를 우회하고 나머지 두 블록/CNN/edge checkpoint는 그대로 유지한다. 모델 깊이를 줄이거나 block 실행을 생략하지 않는다. 기존 `no_outer_l0`는 명시적 CLI 선택으로만 재현 가능하다.

기준3batch와 후보별3batch, 총12회 DEBUG update다. 첫 회는 warm-up이고 각 후보가 OOM이면 오류 문자열만 보존하고 다음 독립 후보로 넘어간다. 실패 traceback의 CUDA tensor를 다음 후보까지 보관하지 않는다. OOM 뒤 작은 batch나 graph로 재시도하지 않는다. 완료하지 못한 batch를 진행률에 성공으로 더하지 않는다.

`--operators`는 baseline 첫 warm-up만 실제 PyTorch CPU/CUDA profiler로 측정해 `baseline_operators.txt`를 기록한다. 나머지 시간 비교 batch는 profiler 없이 실행한다. CUDA event로 L0 forward와 그 외 forward, L0 출력에 gradient가 도착하기 전후 backward 구간도 기록한다. **backward 구간을 배타적인 L1/L2와 L0 시간으로 단정하지 않는다.** autograd scheduling에서 다른 parameter backward가 섞일 수 있다. 원본 loss 함수를 복제하거나 분리된 backward를 실행하지 않는다.

후속 로컬 검증: 6 tests passed,6.292초. 실제 CUDA BF16의3층/128차원/4head synthetic operator, dropout, 다중 edge chunk에서 baseline·전체 해제·각 블록 선택적 해제의 출력/gradient/CUDA RNG가 정확히 일치했다. 선택된 블록만 우회하고 method가 복원되는지, CUDA timestamp/profiler API와 hook 제거도 확인했다. **실제 full-support 서버 후보의 성능·VRAM·모델 전체 parity는 아직 미측정**이다. production runtime/source 파일은 수정하지 않았다.

## 확인된 시간과 후보

사용자가 제공한 A100 MIG 1g.10gb 서버 step29–78: 평균 step22.104초, forward6.199초, backward15.451초, loader0.068초, support-plan0.083초, checkpoint request/wait0.271초, peak allocated5.99GiB. forward/backward 측정 구간 합계가 전체의 약98%다. GPU 포화율이나 L0 단독 비중을 측정한 결과는 아니다. prepare_support의 학습 가능한 L1/L2 계산은 forward/backward에도 포함된다.

현재 L0에는 블록 전체 checkpoint와 내부 attention logit checkpoint가 중첩된다. `no_outer_l0` 후보는 전자만 끄고 후자는 유지한다. CNN checkpoint, message aggregation, L1/L2, precision, graph/edge 순서, 물리적 배치와 loss는 변경하지 않는다. CPU worker 추가나 매 step 파일 저장 생략을 해결책으로 적용하지 않는다.

## 비교 도구

`tools/benchmark_v22_recompute_debug.py`는 정지된 production v5 checkpoint의 source/runtime/cache hash, model/Adam/RNG/plan/cursor, 전체 support binding을 기존 검증기로 확인한다. 현재 epoch의 실제 다음 bucket batch3개를 그대로 사용하며, 다음 epoch로 넘어가거나 동일 batch를 반복해서 채우지 않는다. phase가 optimization이 아니거나 남은 batch가 부족하면 명시적으로 거부한다.

각 방식은 같은 checkpoint에서 시작한다. 첫 batch는 warm-up, 이후 두 batch의 compute 시간을 비교한다. 모델·Adam·전체 memory·현재 episode plan·RNG를 다시 로드한다. gradient는 production clipping 이후 모든 trainable parameter를 비교한다. loss/parts, gradient, 갱신된 model/Adam, CPU/CUDA/NumPy/Python RNG가 모든 검사 batch에서 정확히 일치해야 후보로 남는다. CPU 증거 복사는 CUDA 시간·peak capture 뒤 수행한다. 원래 optimizer checkpoint 저장 비용이나 L2 plan 생성 비용은 이 compute 시간에 포함하지 않으며 `steps.jsonl`에 범위를 기록한다.

candidate OOM은 result.json에 탈락으로 남긴다. GPU/모델/배치를 줄이는 fallback은 없다. 설정된 VRAM 예산 이내이고 정확히 일치하면서 compute 시간이5% 이상 줄어야 `eligible_for_further_validation=true`다. 이는 **추가 검증 후보 판정이며 production 전환 승인이나 모든 worst batch의 메모리 보장이 아니다**. 실행 순서/클록/공유 GPU 간섭과 표본 수의 한계가 있어 큰 개선율을 미리 약속하지 않는다. 아직 production migration 실행기는 추가하지 않는다.

새 파일만 추가하므로 core83/runtime20 provenance는 바뀌지 않는다. 이미 만든 graph cache, support, 저장된 학습을 재생성하거나 새 계약으로 재표시하지 않는다. probe는 DEBUG 결과 JSON만 기록하며 원본 checkpoint에 쓰지 않는다. baseline 비교용 텐서 복사본은 RAM에만 유지한다.

## 서버 실행

현재 학습 화면에서 Ctrl+C를 한 번 누르고 **PAUSED까지 기다린 후** 실행한다. probe는 원본 worker/trainer가 남아 있으면 거부한다. 다른 프로세스 종료나 강제 종료는 수행하지 않는다.

```bash
conda activate nnunet &&
cd /home/aicompetition06/Medical/HierCP-v22-e1e34bf &&
git fetch origin codex/v222-server-r6 &&
git switch --detach FETCH_HEAD &&
python -u tools/benchmark_v22_recompute_debug.py \
  --cache work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --checkpoint work/v22_gnn40_resume114_20260928/training/checkpoint_latest.pt \
  --output work/v22_recompute_selective_20260928_DEBUG \
  --operators
```

원본 실행은 benchmark 후 자동 재개하지 않는다. 후보 결과를 보고 실행 정책 전환을 판단한다. 후보를 적용하지 않고 원래 학습을 재개하려면 기존 process runner에 원본 `--cache`와 위 `--resume` checkpoint를 전달하고 새 output을 사용한다. saved workspace/optimizer/RNG 정책을 유지하며 `--edge-workspace-mib`를 재지정하지 않는다.

## 로컬 검증과 미검증

`test_v22_recompute_probe`: 4 tests passed, 5.014초. 짧은 CUDA BF16 synthetic attention operator(128 width/4 heads, dropout, 여러 edge chunk)의 두 실행에서 출력·gradient·CUDA RNG 정확 일치. CPU snapshot의 Adam/NumPy 별칭 방지, 정책 복원, 실제 bucket cursor 선택과 epoch wrap 거부도 검사했다. 초기 test fixture의 training 속성 누락은 수정 후 재실행했다.

로컬 GPU는 RTX5070Ti이며 조회 시16GB 중5693MiB, utilization60%로 다른 작업이 사용 중이었다. 그 작업을 중단하거나 로컬 full-cohort benchmark를 실행하지 않았다. synthetic CUDA unit은 실제 CT capacity/속도 증거가 아니다. **서버 전체 support A/B, 모델 전체 gradient parity, 실제 개선율, production 전환은 아직 미검증이다.**

## 2026-09-28 추가: 후보를 실행하지 않는 한 step profiler

현재 병목은 해결되지 않았다. 표본/support 수를 줄이는 제안은 적용하지 않는다.
로컬 RTX5070Ti에서 명시적 synthetic DEBUG 연산 검사만 수행했다. 32768 node,
262144 edge, hidden128/head4, BF16, 결정론 설정을 사용했다. 이는 CT 전체 batch
검증이나 MIG 성능 측정이 아니다. 조회 당시 다른 작업이 GPU5716MiB/63%를 사용했고,
CPU16 logical, 가용 RAM35.79GiB였다. 다른 작업은 건드리지 않았다.

- 일반 indexing을 index_select로 대체한 엣지 합산: 출력/feature gradient/attention
  gradient는 6회 모두 bitwise 일치. warm-up 제외 baseline7.67–11.46ms,
  후보9.71–12.39ms로 일관된 개선 없음. 후보 파일 제거, production 미적용.
- 난수를 쓰지 않는 edge checkpoint의 RNG 보존을 생략한 검사: 출력/입력 및 parameter
  gradient/RNG 일치. warm-up 제외 baseline29.37–34.22ms, 후보24.47–38.09ms.
  공유 GPU에서 개선 확정 불가, production 미적용.
- baseline attention profiler에서는 copy18.42%, multiply14.53%, add13.60%,
  index_put11.63%, fill10.29%의 self CUDA 비중을 관측했다. 커널 행은 ATen 행과
  겹치므로 합산하지 않는다. 서버 전체 모델의 연산 비중으로 외삽하지 않는다.

`--profile-only`는 원본 optimization checkpoint의 **다음 full physical batch 하나**와
전체 saved support를 사용한다. baseline forward/backward/clip/optimizer를 별도 모델에서
실행하되 production checkpoint는 쓰지 않는다. 다른 후보/재계산 해제를 실행하지 않으며,
GPU operator 표를 즉시 화면에 출력하고 CPU 표도 파일로 저장한다. cold/profile 오버헤드가
있으므로 처리량 향상 판정에 사용하지 않는다. training이 실행 중이면 먼저 거부한다.

현재 학습 뷰에서 Ctrl+C 후 **PAUSED 확인** 다음 실행한다. 기존 캐시를 다시 만들지 않는다.

```bash
conda activate nnunet &&
cd /home/aicompetition06/Medical/HierCP-v22-e1e34bf &&
git fetch origin codex/v222-server-r6 &&
git switch --detach FETCH_HEAD &&
python -u tools/benchmark_v22_recompute_debug.py \
  --cache work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --checkpoint work/v22_gnn40_resume114_20260928/training/checkpoint_latest.pt \
  --output work/v22_one_step_operators_20260928_DEBUG \
  --profile-only
```

짧은 CUDA/단위검사6개 통과(6.338초). one-step cursor/epoch wrap/미완성 support 거부도
검사했다. 실제 서버 profiler 실행, 전체 모델 후보 parity, 속도 개선, 전체 학습/평가는
이 추가 작업에서 수행하지 않았다. core/runtime/cache contract 변경 없음.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 3batch는 별도 명시된 DEBUG probe다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. saved measured batch/workers를 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 서버 제공 시간 기록과 로컬GPU 상태 확인; probe도 자원을 기록한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. candidate OOM은 탈락 처리한다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. synthetic unit을 실제 CT 결과로 보고하지 않는다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. probe는 기존 production loss/update를 호출한다. 실제 전체 모델 비교는 서버에서 남아 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
