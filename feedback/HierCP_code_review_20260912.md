# HierCP 소스·실행 파이프라인 검토 보고서

검토일: 2026-09-12 KST  
입력: `gpt_handoff.md` / `code.txt`  
소스 스냅샷: `ec2d8388f39c6b03ed32a3677fbdd6eaafed4731`  
검토 방식: 원본 보존, 별도 디렉터리에서 정적 분석 및 명시적 synthetic DEBUG 진단

## 1. 최종 판단

**핵심 모델과 feedback은 이름만 연결한 toy/placeholder가 아니다. L0/L1/L2·CNN·랭킹 학습·raw-target CP·nnU-Net 실제 예측 관측·별도 difficulty GNN·다음 epoch 선택으로 이어지는 구현이 존재한다. 그러나 지금 소스를 “전체 파이프라인 검증 완료”로 승인할 수는 없다.**

우선순위는 다음과 같다.

| ID | 분류·우선순위 | 핵심 판단 | 확인 방법 |
|---|---|---|---|
| F1 | 구조적 위험 / P1 | 원래 종양 위치의 수치 feature를 가려도 source의 실제 region을 알려주는 L1 topology가 남는다. “source 위치 shortcut 차단 완료”라는 포괄적 표현은 과하다. | 실제 graph builder·mask·forward 전달 경로 및 재현 |
| F2 | 확정 운영 결함 / P1 | feedback contract를 최종 파일명으로 먼저 쓴 뒤 검증한다. 늦은 실패 이후 같은 단계 재시도는 기존 파일 때문에 거절된다. | 실제 publisher 함수에 late verifier 실패 주입 |
| F3 | 확정 실행/문서 불일치 / P1, offline 한정 | README의 `run.py full`은 CP arm에서 무조건 거절되는 legacy nnU-Net 경로로 들어간다. 비싼 이전 단계나 baseline 뒤에 실패할 수 있다. | 실제 호출 경로 및 stage 경계 재현 |
| F4 | 자원·성능 위험 / P2 | difficulty GNN의 batch calibration은 prefix만 측정하고 host/cgroup 최악 입력량 guard가 main quality 경로보다 약하다. graph cold build·reload 비용도 크다. | 구현 대조; 서버 peak/속도는 미측정 |
| F5 | 확정 평가 결함 / P2 | threshold 미달 간선 점수가 Hungarian secondary objective에 남아, 같은 TP 수에서 유효 pair quality 및 검출 GT 선택이 의도와 달라질 수 있다. | 실제 evaluator에서 반례 재현 |
| F6 | 재개 기능 제한 / P2 | ordinary fresh run은 recovery/upgrade와 달리 동일 root의 `--resume-experiment`를 지원하지 않는다. | 분기·CLI·journal 연결 대조 |
| F7 | 연구/종결 계약 공백 / P2 | Full/Basic은 quality gate도 다르다. feedback 독립 효과가 분리되지 않으며 현재 driver에 paired evaluation/statistics와 checkpoint→prediction 연결 완료 단계가 없다. | 설정·driver·평가 provenance 대조 |
| F8 | 낮은 우선순위 / P3 | 항상 0으로 마스킹하는 upper raw feature 열에 대응하는 Linear weight 열은 데이터 gradient가 0이다. 전체 L1/L2 미연결은 아니지만 “모든 parameter가 유효 사용”과 다르다. | 실제 mask + Linear backward |

P1은 기존 결과를 지우거나 즉시 재학습하라는 뜻이 아니다. **현재 자산을 보존한 채 결함 경계와 실험 정의를 먼저 확정해야 한다.** F1/F7은 성능 저하나 실제 held-out 오염을 관측한 결과가 아니다.

## 2. 실제 확인 범위와 한계

### 2.1 전달물 무결성

`code.txt`의 159개 FILE/END FILE 경계를 분리했다. Python은 134개, JSON은 5개, 테스트 모듈은 65개다. 전체 텍스트는 약 6.8만 줄이다. 본문 LF와 export separator를 구분해 원문 byte를 복원했으며, 별도 `gpt_handoff.md`는 snapshot 내부의 같은 파일과 byte-identical이다.

이것은 **첨부 snapshot 내부의 완전성** 확인이다. 원격 Git checkout의 현재 파일 목록이나 사용자 서버 상태를 확인한 것은 아니다. snapshot에 없는 실제 의료 cohort·trained checkpoint·bank·journal·서버 로그는 이 검토에서 검증하지 않았다.

| 검사 | 이번 결과 | 의미 |
|---|---|---|
| 159개 파일 인벤토리 | 경계·이름·본문·SHA 등록 | 파일 누락 없이 검사 대상으로 관리 |
| Python 구문 | 134/134 통과, Python 3.10 grammar 기준 AST | 외부 라이브러리 호환성·수치 정확성 증명 아님 |
| JSON 구문/중복 key | 5개 정상, 중복 key 없음 | 스키마 의미는 별도 대조 |
| 내부 import/name 연결 | 940개 정적 참조 해석, 경고 0 | 동적 import/alias/외부 API는 전역 증명하지 않음 |
| 중복 정의·literal dict key·명백한 내부 keyword 불일치 | 해당 정적 검사 경고 0 | 모든 호출이 실행된 것은 아님 |
| custom trainer/helper SHA | 10/10 일치 | installer MODULES·SHA256SUMS·복원 source bytes 일치 |
| 새 진단 | 8/8 진단 시나리오 확인 | 결함 재현 성공도 포함. 프로젝트 무결점 8/8이라는 뜻 아님 |
| 원본 보존 | 복원 159개 파일 SHA 재검사 모두 일치 | 감사가 소스 패치를 적용하지 않음 |

파일별 대장은 동반 `HierCP_review_inventory_20260912.md`에 있다. **전수라는 말은 인벤토리·구문·정적 참조를 뜻한다. 159개 모든 파일의 모든 분기를 같은 깊이로 수동 증명하거나 native 실행했다고 주장하지 않는다.** 핵심 활성 경로와 복잡한 수치·상태 경계는 아래 별도로 대조했다.

### 2.2 실행 환경과 회귀 테스트

이번 환경은 Python 3.13.5, PyTorch 2.10.0+cpu, CUDA 없음이다. PyG, nnU-Net, NiBabel 및 native augmentation 관련 의존성이 갖추어져 있지 않다. 사용자 서버 환경을 대신하지 않는다. 로컬 자원은 CPU affinity 5, cgroup 메모리 약 4 GiB로 확인했으며, 사용자 서버의 GPU/VRAM/CPU/RAM 값은 추정하지 않았다.

기존 unittest discovery를 실제로 시도했다. `testsRun=405`, 정상 성공 callback 344개, error event 52개, skip event 22개, assertion failure event 0개가 기록됐다. 실행기 본문 약 6.13초, discovery 포함 약 8.14초다. **setup/subtest/module-load 이벤트 때문에 이 숫자는 서로 배타적인 405개 분할이 아니다.** 전체 suite는 실패로 끝났고 통과로 승인하지 않았다.

오류의 상당수는 누락된 PyG/native 의존성이다. 일부는 import/patch 경계의 `AttributeError`도 포함한다. 이를 “제품 결함 52개”로 계산하지도, 모든 오류가 반드시 환경 탓이라고 일괄 면책하지도 않았다. 원본 로그와 개별 test ID를 evidence bundle에 보존했다.

handoff의 과거 662개 suite/659개 통과나 58개 중 57개 통과는 **기존 작성자의 과거 실행 기록**이다. 이번 실행 결과와 합치지 않았다. production-sized optimizer smoke, 실제 전체 cohort bank, 40-epoch quality GNN, 250-epoch nnU-Net, GPU resume, full paired evaluation은 실행하지 않았다.

## 3. 실제 파이프라인 구조

### 3.1 현재 online-feedback 활성 경로

```text
원본 CT/label 및 case-ID outer split
  ├─ outer validation: GNN fitting / donor / recipient / feedback에서 제외
  └─ outer train
       ├─ inner train → region/context prototype fitting + quality GNN 학습
       └─ inner validation → quality checkpoint 선택

검증된 quality checkpoint + 공통 train-only nnU-Net planning / 고정 전처리
  → raw 후보마다 실제 CT/label paste를 표현하는 typed bank
  → frozen quality score + 128개 raw 후보 + source slot 보존
  → feedback contract 출판·검증
  → Full nnU-Net 250 epochs
       정상 forward → 최고해상도 pre-update 예측을 detach하여 CP 오차 관측
       정상 segmentation loss/backward/optimizer는 그대로 실행
       epoch 끝 → EMA table + 별도 difficulty GNN update
       다음 epoch → frozen difficulty snapshot으로 후보 선택
  → Basic control nnU-Net 250 epochs
       동일 event/source/jitter RNG 규칙, 후보는 전체 hard-valid pool에서 uniform
  → 현재 driver 종료
       별도 paired evaluation/statistics/결과 provenance 종결 필요
```

Quality GNN과 difficulty GNN은 같은 계층 구조를 사용하지만 목적과 상태가 다르다. quality score/bank를 feedback으로 덮어쓰는 구조가 아니다. 또한 이것은 **end-to-end differentiable segmentation-quality optimization이 아니라**, 사전 placement-ranking + 학습 중 관측을 이용한 별도 difficulty 학습이라는 두 단계 연구 구조다.

현재 driver: `tools/run_feedback_experiment.py:676–750` / `code.txt:66212–66286`. 실제 nnU-Net launcher: `tools/train_online_feedback.py:20–93` / `code.txt:67157–67230`.

### 3.2 설정 규모를 정확히 읽어야 하는 항목

| 항목 | 첨부 설정/계약 |
|---|---|
| schema / geometry / architecture | full_v22 / level0_physical_closure_v2 / hiercp_conditioned_readout_v3 |
| GNN hidden / heads / blocks | 128 / 4 / L0 3, L1 2, L2 2 |
| CNN 입력 / channels | 48³ / 12, 24, 32 |
| patient regions / population prototypes | 24 / 16 |
| quality ranking 후보 / placement 후보 | 8 / 128 |
| quality 학습 / 각 nnU-Net arm | 40 / 250 epochs |
| context 384 | 최종 노드 cap이 아니라 context seed 수 |
| physical batch / workers | quality 경로는 auto 측정; accumulation 1; target effective batch null |
| raw source padding / clearance | 2 voxels / 2 voxels |
| raw center separation | 12 native voxels, 추가 physical separation 0 |
| liver coverage / paste | 0.85 / hard paste |
| HU jitter | scale 0.95–1.05, shift −5–+5 |
| CP probability | 0.5 |
| online eligible source 크기 | 기본 nnunet config의 equivalent diameter 0 초과, 20 mm 이하 |
| difficulty mixing | exploration 20%, easy retention 20%, stage band 60% |
| difficulty 통계 | EMA α=0.25, minimum observations 1, stale 25 epochs |
| predicted difficulty freshness | maximum age 1 epoch |

`config/nnunet.json`의 small-tumor 조건은 명시적인 실험 설계다. “모든 source”는 **모든 eligible source**이지 모든 크기의 병변을 제한 없이 포함한다는 뜻이 아니다. equivalent diameter가 작아도 길쭉한 병변의 bounding box가 native patch보다 클 수 있다.

`48³`는 GNN dense encoder 입력 크기다. 전체 종양 geometry까지 48³에 잘라 버렸다는 뜻이 아니다. `sample_context_nodes=384` 역시 interface/hop closure를 포함한 최종 local graph 크기와 다르다. 이런 숫자를 서로 바꾸어 해석하면 규모를 잘못 진단하게 된다.

## 4. F1 — 수치 위치를 숨겨도 L1 topology가 source의 실제 region을 알려준다

**분류: 구조적으로 확인된 shortcut 가능성. 실제 학습된 score가 이를 이용했다는 주장은 아님.**

관련 코드:

- `hiercp/hierarchy.py:410–465` / `code.txt:16746–16801`
- `hiercp/model.py:161–187` / `code.txt:17903–17929`
- `hiercp/model.py:869–915` / `code.txt:18611–18657`
- `hiercp/curriculum.py:205–222` / `code.txt:13631–13648`
- `tools/causality.py:1271–1300` / `code.txt:46442–46471`

현재 차단은 tumor/candidate raw feature 일부 열과 source-tumor 관련 edge attribute의 위치·거리·same-region 열을 0으로 만드는 방식이다. 그러나 graph builder는 다음을 계속 만든다.

```text
tumor.region_index = region_at(source.anchor_center)
tumor --hosted_by--> 실제 source region
region --hosts_tumor--> tumor
candidate --belongs_to--> 후보의 region
region --contains_candidate--> candidate
```

`PatientRegionPyGEncoder.forward_raw`는 마스킹한 edge attribute를 쓰지만 **`raw_batch.edge_index_dict`는 그대로 message passing에 전달한다.** 따라서 `tumor → source region → candidate`라는 2-hop 경로 자체가 source region과 같은 후보를 구별할 수 있다. region feature에는 위치/context도 남아 있다.

이것이 중요한 이유는 quality pretext의 positive가 `source.anchor_center`로 만들어지고, source region/prototype을 기준으로 negative category도 나뉘기 때문이다. **context 호환성을 학습해야 하는 문제를 원래 주소와의 관계로 부분적으로 풀 수 있는 정보 경로가 남아 있다.**

새 DEBUG probe는 실제 graph builder와 mask 함수를 실행했다. PyG container와 feature 추출만 명시적 경계 double로 대체했다. 마스킹 후에도 source region 1이 복원됐고, 두 후보의 source-region 2-hop 연결은 `[True, False]`로 구별됐다. 실제 PyG forward나 서버 checkpoint 점수 변화는 측정하지 않았다.

기존 `upper_position_noise` causality 검사는 `raw_x`, `pos`, `edge_attr`를 바꾸지만 이 source-host topology를 바꾸지 않는다. 그러므로 그 검사가 통과해도 **모든 형태의 source 위치 shortcut이 차단됐다는 결론은 나오지 않는다.**

**수정 방향:** source의 생물학적/context 정보와 “정답이 있었던 region ID/주소”를 별도로 정의해야 한다. 허용하지 않을 topology라면 별도 architecture version에서 제거/재표현하고, source-host rewiring·동일-region 후보·좌표/attribute/topology 분리 perturbation을 추가한다. 기존 bank/checkpoint의 schema만 바꾸어 새 설계로 승인하면 안 된다. 이 검토는 L1 전체 제거를 권하지 않는다.

## 5. F2 — 계약 출판 후 검증 실패가 지원되는 재시도를 막는다

**분류: 확정 운영 결함. 현재 feedback 경로에 적용.**

Publisher: `tools/online_cp_curriculum.py:23–114` / `code.txt:60100–60191`. 단계 재개: `tools/run_feedback_experiment.py:378–435` / `code.txt:65914–65971`.

처리 순서는 다음과 같다.

```text
feedback_contract.json이 있으면 FileExistsError
 → 최종 이름으로 open("x")
 → JSON 기록
 → verify_curriculum_bank_contract()
 → 성공 후에야 stage completion 기록
```

늦은 verifier 실패, I/O 오류, 기록 이후 완료 journal 이전의 중단이 발생하면 **최종 이름의 계약 파일은 남지만 해당 단계는 미완료**다. 다음 재개에서는 같은 publisher 명령을 다시 실행하고, 이번에는 파일이 있다는 이유로 거절한다. `_resume_command`가 특별히 다루는 것은 quality training과 Full/Basic native resume이며, contract publication에는 해당 복구 처리가 없다.

새 probe에서 실제 `publish()`의 앞쪽 native 완료 경계는 명시적으로 double 처리하고 마지막 verifier에 `OSError`를 주입했다. 첫 시도는 `OSError`, 최종 파일 잔존, 두 번째 시도는 `FileExistsError`였다. 남은 파일 byte도 바뀌지 않았다. 실제 의료 데이터를 승인한 테스트가 아니라 **출판/실패/재시도 상태 전이 재현**이다.

**수정 방향:** 임시/staging 계약을 검증한 뒤 최종 이름으로 atomic publish해야 한다. 이미 있는 파일은 삭제하거나 무조건 덮어쓰는 것이 아니라, 같은 입력/계약/현재 검증을 모두 통과할 때만 idempotent reuse하거나 명시적인 owned-incomplete recovery 절차를 제공해야 한다.

회귀 검사에는 다음 경계를 모두 포함해야 한다: write 전 실패, JSON write 중 실패, write 후 verify 실패, verify 성공 후 journal 전 실패, 정상 동일 재시도, 실제 SHA가 달라진 재시도 거절.

### 같은 종류의 추가 복구 창

`hiercp/feedback.py:195–226` / `code.txt:14670–14701`의 graph cache도 `.pt`를 최종 이름으로 쓰고 `.json` receipt를 뒤에 쓴다. 중단 후 한쪽만 남으면 이후 `_canonical()`이 명시적으로 거절한다. 이는 손상 cache의 조용한 재사용을 막는 장점이 있으나, **안전한 incomplete artifact 회수 절차가 없으면 native checkpoint resume가 graph cache에서 다시 막힐 수 있다.** 이 부분은 설계상 fail-closed인 것과 재개 가능성을 구분해 보완해야 한다.

## 6. F3 — README의 standalone full은 현재 코드에서 완주할 수 없다

**분류: 확정 실행/문서 불일치. current feedback과 구분되는 legacy/offline 경로.**

- `run.py full`: `run.py:813–820` / `code.txt:27936–27943`
- offline nnU-Net gate: `tools/nnunet.py:480–519` / `code.txt:51822–51861`
- README: standalone production workflow의 `all`/`full` 설명.

`run.py full`은 prepare/train/generate/validate/assemble 다음에 `tools.nnunet all`을 호출한다. 그 안의 `train()`은 각 fold에 대해 baseline을 먼저 처리하고 `hierarchical_copy_paste` arm을 호출한다. 그런데 `train_one()`은 그 CP condition이면 함수 초반에 **항상 예외**를 낸다. globally generated CP가 held-out label을 GNN fitting/선택에 사용하지 않았음을 증명할 수 없다는 이유다.

이 보호장치는 타당한 누수 방지 경계다. 문제는 **완료 가능한 정식 workflow인 것처럼 README와 top-level target이 남아 있고, 거절 시점이 뒤쪽이라는 점**이다. 준비가 정상이라면 expensive GNN/offline 생성 또는 baseline 실행 뒤에야 실패할 수 있다.

새 probe는 baseline을 실행하지 않는 경계 recorder로 두고 실제 `train`/gate 정의를 실행했다. 호출 순서는 `baseline → hierarchical_copy_paste → 명시적 예외`였다.

**수정 방향:** leakage guard를 제거하지 않는다. unsupported target은 최상위 preflight에서 비용 발생 전에 거절하거나, 실제로 지원되는 fold-specific 경로에 맞게 별도 target과 문서를 정리한다. 이미 만들어진 데이터/결과를 지우지 않는다.

## 7. F4 — difficulty GNN의 자원 측정과 graph I/O는 전체 cohort에서 다시 검증해야 한다

**분류: 구현상 확인된 자원 정책 차이와 병목 가능성. 서버 OOM/처리량을 측정한 결과 아님.**

### 7.1 calibration 대표성이 약하다

`hiercp/feedback.py:477–579` / `code.txt:14952–15054`의 `_calibrate()`는 각 physical batch 후보에 대해 `entries[:size]`를 측정한다. 실제 update는 이후 전체 관측 entry를 shuffle하여 사용한다. 그래프 크기가 entry마다 다르면 prefix 통과는 이후 더 큰 batch 조합의 안전성을 보장하지 않는다.

main quality 경로는 `hiercp/training_resources.py:1–111` / `code.txt:26959–27069` 및 `hiercp/pipeline.py`의 calibration에서 전체 cache 입력량·host budget·큰 입력 순서를 다루지만 feedback 쪽에는 같은 cgroup 기반 사전 host budget 경계가 없다. CUDA OOM만 직접 잡고 host available RAM은 뒤에서 보고한다. 대형 source의 graph load/collate에서 host 메모리가 먼저 부족하면 이 CUDA 경계에 도달하지 못할 수 있다.

이 코드는 학습 데이터를 실제로 prefix만 쓰는 축소 버그는 아니다. **측정 입력의 대표성과 승인 범위 문제**다. 보고서에도 `representative_calibration_only=True`가 있으므로 현재 구현이 스스로 전체 안전성을 증명한다고 쓰는 것은 아니다.

### 7.2 cold graph build와 재로딩 비용

`hiercp/feedback.py:195–287` / `code.txt:14670–14762`을 보면 새 source entry의 graph를 만들 때 raw case를 다시 읽고 `load_or_build_patient_regions(..., cache_dir=None)`를 호출한다. 같은 환자의 여러 source component에도 기존 검증된 region cache를 직접 공유하지 않는다. candidate graph는 캐시되지만, 그 전에 case/region/거리 변환 준비가 반복될 수 있다.

이미 cache가 있어도 `_canonical()`은 graph 파일 전체 SHA를 확인하고 `torch.load(..., map_location="cpu", weights_only=False)`로 읽는다. 이 호출에는 mmap이 없다. 실제 source는 128개 후보의 graph payload를 포함하므로 quality 학습의 8개 후보 sample과 비용이 같다고 볼 수 없다. 다만 chunking·graph 공유가 있으므로 이를 근거 없이 16배 RAM이라고 환산하지 않는다.

또 calibration의 매 repeat마다 `_loader()`를 새로 만들고, update와 prediction도 각각 재측정한다. worker 시작/파일 검증/cold I/O가 throughput 측정에 들어간다. 순수 forward/backward 속도만 측정한 수치가 아니다.

**수정 방향:** verified patient-level region/cache 공유, cold/warm 준비시간 분리, graph input byte/노드/엣지 전체 분포, 큰 source가 섞인 batch 조합, cgroup host RAM·worker/prefetch·spool 예산을 함께 검증한다. mmap/witness와 측정 결과 재사용은 actual identity가 같을 때만 적용한다. **모델 깊이·48³·128 후보·hop/radius를 줄이는 것이 첫 해법이 아니다.**

## 8. F5 — 평가기의 threshold 미달 간선이 유효 pair 선택에 영향을 준다

**분류: 확정 평가 secondary matching 결함. total maximum TP와 구분.**

관련 함수: `tools/online_eval_v2.py:377–405` / `code.txt:60894–60922`. 선택된 GT의 크기별 기록: `tools/online_eval_v2.py:1094–1154` / `code.txt:61611–61671`.

현재 quality-aware matching은 다음 objective를 쓴다.

```python
valid = (intersections > 0) & (score >= threshold)
objective = valid.astype(float) * bonus + np.clip(score, 0.0, 1.0)
```

`bonus`가 충분히 커서 **유효 one-to-one TP 개수를 우선 최대화하는 것**은 맞다. 그러나 `np.clip(score, ...)`는 invalid edge에도 남는다. 같은 TP 개수라면 최종적으로 버려질 invalid edge의 점수까지 포함해 matching을 고른다. 따라서 주석의 “그 다음 pair quality”가 **최종 유효 pair의 quality**를 뜻한다면 구현이 맞지 않는다.

실제 evaluator 함수에 아래의 가능한 component-count matrix를 넣었다.

```text
GT voxel 수       = [100, 200]
Prediction voxel 수 = [100, 200]
Intersection      = [[45, 36],
                     [45,  0]]

Dice              = [[0.45, 0.24],
                     [0.30, 0.00]]
Threshold         = 0.25
```

각 행/열 intersection 합은 그 component 크기를 넘지 않는다. 두 matching 모두 TP 1이 가능하다.

- 유효 pair만 보면 GT1–P1의 Dice 0.45가 더 좋다.
- 현재 objective는 GT1–P2의 **invalid 0.24**와 GT2–P1의 valid 0.30을 더해 0.54로 평가한다.
- 실제 결과는 GT2–P1의 0.30이 선택됐다.

**전체 TP/FP/FN이 이 반례에서 틀린 것은 아니다.** 달라지는 것은 어떤 GT를 detected로 기록하는지와 해당 pair metric이다. GT1/GT2가 서로 다른 크기 구간이면 size-specific sensitivity에도 영향이 갈 수 있다. 반면 whole-volume Dice와 별도 `max_dice_match()` 계산을 이 결함에 묶어서 잘못됐다고 말하면 안 된다.

**수정 방향 예시:** secondary score도 valid mask 내부에서만 계산한다.

```python
secondary = np.where(valid, np.clip(score, 0.0, 1.0), 0.0)
objective = valid.astype(np.float64) * bonus + secondary
```

이는 감사 보고서의 제안이며 소스에 적용하지 않았다. 적용 시 evaluator version/metric definition을 바꾸고, 기존 예측을 사용해 **새 출력 폴더에서** 재평가해야 한다. 과거 결과를 덮어쓰거나 두 정의의 통계표를 섞으면 안 된다. 위 반례와 row/column 순열·다중 GT 크기구간 회귀 검사가 필요하다.

## 9. F6 — fresh 실행의 재개 경로가 recovery/upgrade와 다르다

`tools/run_feedback_experiment.py:676–750` / `code.txt:66212–66286`와 `tools/run_feedback_experiment.py:777–780` / `code.txt:66313–66316`.

ordinary fresh branch는 private runtime을 만들고 command를 순서대로 실행하지만, recovery/upgrade branch와 같은 stage journal 기반 재개 driver를 사용하지 않는다. 동시에 기존 root가 있으면 fresh 재실행을 거절하며 `--resume-experiment`는 원래 `--recover-from` 또는 `--upgrade-bank-from`을 요구한다.

즉 **“기존 experiment 전체는 언제든 --resume-experiment로 이어진다”는 설명은 틀리다.** 지금 handoff의 특정 upgrade 경로는 지원되지만 일반 fresh 경로는 같은 방식이 아니다. 원격 프로세스·lock·완료 증거가 없는 상태에서 새 명령을 반복하는 것도 위험하다.

**수정 방향:** fresh도 같은 durable stage machine을 사용하도록 통일하거나, 현재 지원되는 fresh failure recovery 절차와 한계를 명확히 제공한다. incomplete runtime, bank, checkpoint를 임의 승인하는 fallback을 만들면 안 된다.

## 10. F7 — 실험 효과의 분리와 결과 종결 계약

### 10.1 Full/Basic 차이는 feedback 하나가 아니다

`custom_trainers/onlinecp_feedback_policy.py:360–401` / `code.txt:5877–5918` 및 `config/online_cp_feedback.json`.

Full은 frozen quality gate를 먼저 적용하고 measured/predicted difficulty 기반 curriculum을 사용한다. Basic은 전체 hard-valid 128개에서 uniform이다. event/source/jitter draw를 paired하게 유지해도 **quality-gate 적용 여부와 difficulty sampling 두 요소가 함께 달라진다.**

따라서 Full이 더 좋다는 결과만으로 “difficulty GNN/feedback의 독립 기여”를 확정할 수 없다. 그 효과를 분리하려면 같은 quality gate의 uniform control 등 사전 정의된 대조 실험이 필요하다. 새 대조군·fold·추가 학습은 이번 감사가 승인하거나 실행한 범위가 아니다.

### 10.2 새 feedback 결과의 end-to-end 종결이 아직 필요하다

현재 driver는 training command 뒤에 downstream comparison/statistical evaluation을 하지 않았다고 출력한다. `tools/online_eval_v2.py`는 explicit `--basic-trainer`/`--hier-trainer`를 지원하므로 **평가 코드가 아예 없거나 feedback 결과를 절대 읽을 수 없다는 뜻은 아니다.**

다만 generic evaluation provenance는 prediction/GT/cohort/metric bytes를 묶고, original training-checkpoint linkage는 미검증이라고 명시한다: `tools/online_eval_v2.py:1531–1543` / `code.txt:62048–62060`. 현재 feedback checkpoint/bank/epoch receipt → native prediction 생성 identity → paired report → 최종 completion receipt의 닫힌 연결이 별도로 필요하다.

외부 nnU-Net training 함수의 validation 단계에서 예측을 만들 수 있는지와 해당 서버에서 실제 파일이 생성됐는지는 이 snapshot만으로 확정하지 않는다. “별도 paired evaluation을 orchestration하지 않는다”와 “어떤 예측 파일도 존재하지 않는다”는 다른 주장이다.

### 10.3 case-ID 경계와 실제 환자 경계

`validate_nested_cohorts()`와 installed trainer는 exact case-ID 집합을 비교한다. source/prototype/GNN train/held-out을 단순히 섞는 경로를 그대로 둔 구조는 아니다. 그러나 서로 다른 case ID가 같은 사람의 반복 촬영이면 이 검사만으로 환자 단위 독립성이 증명되지 않는다. 환자 grouping 자료는 첨부에 없다.

### 10.4 surviving-support 난이도와 전체 병변 난이도

metric은 crop/augmentation 이후 surviving CP support를 측정한다. 이는 명시적 정의다. 동시에 crop 밖의 별도 component가 없어졌는지 전체 원본 병변과 비교하지 않고, 전달된 support의 boundary contact로 보수적으로 제외한다. 새 진단에서 내부 27-voxel support는 available이었다. crop 밖에 다른 disjoint native component가 있다고 가정해도 그 정보는 metric 입력에 없다.

따라서 `raw_source_voxels`, `native_support_voxels`, `crop_support_voxels`, augmentation 상태를 같이 읽어야 한다. available을 “원본 전체 병변이 완전 보존됨”으로 확대 해석하지 않는다. **이 점은 명시된 surviving-support metric 자체를 틀렸다고 판단한 것이 아니다.**

## 11. F8 — 부분 dead weight와 gradient 검증의 해석

`_mask_upper_shortcuts()`가 raw columns `[0, 1, 2, 4]`를 항상 0으로 만들지만 projection은 기존 raw width를 유지한다. 이 입력에 대응하는 Linear weight 열의 data gradient는 0이다. 실제 mask와 Linear backward로 확인했다.

main quality connectivity 기록은 ``hiercp/pipeline.py:2024–2077` / `code.txt:21588–21641``의 `parameter.grad is not None` 기준으로 연결 이름을 누적한다. 이 조건은 전체 tensor가 graph에 참여하는지는 검사할 수 있지만 **tensor의 모든 열이 유효 신호를 받는다는 증명은 아니다.**

낮은 우선순위의 정리 항목이다. 전체 L1/L2가 끊겼다는 뜻도 아니고 이를 이유로 현재 가중치를 몰래 자르라는 뜻도 아니다. 향후 versioned projection에서 허용 열만 선택하고 compatibility migration/재학습 필요성을 분리해 판단하면 된다.

## 12. 복잡한 부분에서 확인한 정상 연결

### 12.1 L0 geometry, full shape, closure

활성 full_v22에서는 `spatial.py`의 physical footprint 경로를 사용한다. anisotropic spacing을 고려한 transform과 확장된 bbox/ROI를 사용하며, 과거 fixed patch 구현의 결함을 최신 활성 경로에 그대로 붙이면 안 된다. resource limit은 explicit error/진단으로 처리하고 조용한 후보·노드 삭제와 구분한다.

`sample.py`는 context seed에 interface 필수 노드와 지정 hop closure를 합친다. 선택된 node의 관계를 조용히 top-k로 줄여 결과를 바꾸는 식이 아니다. `sample_context_nodes=384`가 최종 전체 노드 384를 뜻하지 않는다.

### 12.2 GNN / CNN / conditioned readout

`model.py`에는 실제 3D CNN, heterogeneous GATv2, learned compatibility sigmoid가 있다. normalized attention만 있는 singleton-neighbor 관계가 학습되지 않는 문제를 compatibility gate로 다룬다. L1의 최종 region/lesion/liver state와 L2의 최종 prototype state는 candidate-conditioned readout에 연결된다. 과거 “마지막 상태가 전혀 사용되지 않는다”는 결론을 그대로 적용하지 않는다.

ablation은 비활성 branch와 입력 폭을 명시적으로 다루며, 현재 full model에서 random output이나 constant score를 반환하는 대체 경로를 핵심 구현으로 확인하지 않았다.

### 12.3 batching 및 streaming

가변 크기 `HeteroData`의 disjoint-union batching과 masked B×C ranking loss를 사용한다. local candidate chunking은 전체 후보를 순서대로 처리해 읽는 계산 방법이며 후보 수를 줄이는 정책이 아니다.

`_StreamedEdgeAggregation`을 실제 CPU float64 autograd로 검사했다. 직접 index-add와 forward 최대 절대 오차 0, `gradcheck=True`, empty-edge 처리 정상이다. 이는 그 연산자의 증거이며 전체 GAT/PyG·GPU chunk numerical equivalence를 대신하지 않는다.

### 12.4 quality 학습 목표와 checkpoint 선택

quality pretext는 source 원위치 positive, easy/inter-region negative, source와 가까운 region/prototype에서는 explicit corruption을 쓰는 구성이다. CE/pairwise/ordinal/mining과 view consistency가 loss→backward→optimizer에 연결된다. validation은 고정 policy를 쓰고 best MRR/accuracy/margin/loss ordering과 last-epoch resume sidecar를 구분한다.

다만 pretext quality와 downstream segmentation difficulty는 다르며, F1의 privileged topology와 별개로 negative label 자체가 병리학적 ground truth라는 의미는 없다.

### 12.5 split·prototype·전처리

prototype fitting은 inner training cases에 묶이고 GNN inner split은 outer train 안에 있다. online planning 경로는 outer-train-only raw view에 fingerprint/plans를 fitting하고 고정된 계획으로 전체 cohort를 transform한다. validation 환자를 CP donor/recipient/feedback으로 쓰지 않는다는 계약을 code와 live key 검증 양쪽에서 확인한다.

단, source snapshot의 guard 존재와 실제 서버 artifact가 그 guard를 통과했다는 사실은 다르다.

### 12.6 raw-target CP 수학

source mask를 원래 위치에서 한 번 resample한 뒤 이동하지 않고 각 raw target의 CT·전체 label-mixture를 기준으로 표현하는 방향은 맞다. 원래 위치에서 작은 donor가 0 voxel이 됐다는 것만으로 다른 target도 불가능하다고 결론 내리면 안 된다.

HU jitter → percentile clipping/mean/std normalization → native resampling 순서를 유지하고, cubic prefilter의 전역 tail을 delta operator로 보존한다. original baseline의 unclipped float64와 target별 변경 extrema를 이용해 native clipping 범위를 다루며, separate-axis는 지원한 kernel만 허용한다.

독립 CPU probe는 raw shape `(7,9,11)`에서 `(10,6,13)`으로 변환하는 global cubic delta와 실제 skimage global resize를 비교했다. crop 최대 절대 오차는 약 `2.22e-15`; full label-mixture는 정확 일치했다. **native nnU-Net prepare_case, 전체 patient volume, 모든 separate-z 조합을 이번에 다시 실행한 결과는 아니다.**

### 12.7 작은 병변/큰 bbox/no-placement

native support가 0인 유효 raw CP도 원래 source/candidate draw를 유지하고 segmentation 학습을 진행한다. 해당 difficulty 관측을 unavailable로 분리한다. 큰 source bbox가 고정 nnU-Net crop보다 크다고 source를 삭제하지 않고 full/native/crop support를 별도 기록한다.

`retain_original`은 exhaustive search에서 실제 후보 0을 증명한 source slot을 유지하는 정책이다. 후보가 1–127개로 부족하거나 검색이 실패한 경우까지 0 후보로 바꾸어 complete를 만들지 않는다.

### 12.8 storage·typed contract·preservation

case baseline/operator는 NPY mmap, candidate metadata/private arrays는 JSON/NPZ, 동일 donor CT/mask는 content-addressed 공유 NPY로 저장한다. pickle 없는 raw bank와 trusted PyTorch checkpoint는 구분된다. 같은 `hiercp_online_bank_v2` envelope라고 legacy/raw mapping을 호환한다고 보지 않는다.

startup의 full SHA 검증 witness를 worker에 전달하고 stat 변경은 계속 확인한다. 이 구조는 매 worker가 같은 대형 baseline을 모두 다시 hashing하는 비용을 줄인다. 다만 F2처럼 fail-closed와 atomic recovery는 별개의 품질 속성이다.

### 12.9 실제 nnU-Net 관측과 epoch 경계

`FeedbackLossObserver`는 정상 forward의 최고해상도 logits를 detach해서 optimizer update 전 CE/boundary/adjacent FP를 읽는다. base segmentation loss를 반환하므로 GNN 학습이 nnU-Net의 본래 loss/backward를 대체하지 않는다. 별도 segmentation forward를 추가하는 설계도 아니다.

available 상태만 records에 넣고, epoch 말에 EMA/difficulty GNN을 갱신한다. 다음 epoch의 frozen snapshot과 worker 재시작으로 이전 prefetch가 다음 epoch policy에 섞이지 않게 한다. checkpoint는 sampler table, GNN/optimizer, observations/prediction provenance, nnU-Net state/RNG/epoch를 함께 저장하도록 연결된다. 실제 native replay는 미검증이다.

### 12.10 최근 bank-upgrade source-history 수정

`feedback_bank_upgrade._check_source_history()`는 input hash가 없는 모든 행을 허용하는 것이 아니다. 구형 실패 행에 한해 뒤의 동일 단계 completed row, input SHA, native output evidence, original recovery receipt를 검증하는 구조다. source journal을 수정해 승인하지 않는다. 관련 DEBUG 26개 성공 callback을 확인했다.

하지만 실제 원격 journal은 첨부에 없으므로 사용자의 최근 오류가 그 누락 hash 조건인지 실제 설정 변경인지 이번에도 확정하지 못한다. “서버 문제 해결 완료”라고 쓰면 안 된다.

### 12.11 평가와 결과 보존

새 evaluator는 기준별 one-to-one detection, whole-mask Dice, lesion quality, 환자 cluster bootstrap 및 paired swap을 구분하고 undefined metric을 무작정 0으로 대체하지 않는다. 새 output을 요구하고 prediction/GT/definition SHA를 묶는 점은 적절하다. F5는 그중 thresholded matching의 secondary choice에 관한 별도 결함이다.

과거 fold-0 exact-argmax의 Full/w/o-L2 비교는 현재 actual-feedback의 효과를 대신하지 않는다. 기존 L2 성능 숫자만으로 L2를 제거하거나 새로운 결과로 재표기할 근거는 없다.

## 13. 추가 진단 8개: 무엇을 실행했는가

| 진단 | 실제 결과 | 주장 가능한 범위 |
|---|---|---|
| source_region_topology_retained | source-region topology 유지, 후보별 2-hop 차이 확인 | 실제 graph builder/mask/forward 전달의 구조적 정보 경로. 학습 score 효과 아님 |
| publisher_failed_verification_retry | OSError→최종 파일 잔존→FileExistsError | 실제 publisher 상태 전이. native 완료 검증은 boundary double |
| offline_full_late_guard | baseline 경계 후 unsupported CP 예외 | 실제 orchestration/gate. baseline 학습은 실행하지 않음 |
| streamed_attention_sum_gradient | forward 오차0, float64 gradcheck 통과 | 실제 custom autograd operator의 CPU 검사 |
| raw_cubic_operator_and_label_mixture | max abs ~2.22e-15, label exact | actual helper vs installed skimage, synthetic CPU |
| masked_projection_columns | [0,1,2,4] data gradient 0 | 실제 mask/Linear backward. 전체 hierarchy 미연결 주장 아님 |
| surviving_support_metric_scope | 내부 27 voxels는 available | crop-local surviving metric의 범위 설명 |
| evaluation_valid_pair_tie_break | valid0.30 선택, same-TP 대안0.45 | 실제 evaluator의 secondary matching 반례 |

진단 script는 original source를 수정하지 않고 필요한 함수 정의를 읽어 실행한다. 경계 double을 쓴 경우 script와 JSON에 명시했다. 환자 데이터를 dummy로 대체해 최종 성능을 보고한 것이 아니다.

## 14. 수정 우선순위와 승인 기준

**1단계 — 값싼 운영 결함부터:** F2 atomic/idempotent contract publication과 interrupted graph cache recovery를 고친다. F3 unsupported offline full을 preflight에서 차단한다. 사용자 데이터·journal·bank·checkpoint를 삭제하는 우회는 금지한다.

**2단계 — 평가 정의를 고정:** F5 valid-only secondary objective와 threshold/size-bin 회귀를 추가하고 evaluator version을 분리한다. 기존 예측은 재사용 가능하지만 원래 보고서를 덮어쓰지 않는다.

**3단계 — source 위치 정보 계약:** F1을 feature와 topology로 나누어 검증한다. 기존 source context를 어디까지 허용할지 결정한 후 versioned graph/model을 만든다. source-host relation을 무조건 제거하는 자동 패치는 하지 않는다.

**4단계 — 실제 하드웨어에서 자원 검증:** Full nnU-Net이 GPU에 올라온 상태로 difficulty GNN의 가장 큰/작은 source와 혼합 batch, cold/warm graph preparation, RAM/VRAM/I/O/worker를 측정한다. 통과한 작은 prefix만으로 cohort 전체 안전을 승인하지 않는다.

**5단계 — native 통합:** 실제 plans/기존 nnU-Net 설치 복사본에서 raw candidate native baseline, tiny zero-support, large partial crop, augmentation의 label/support alignment, 실제 segmentation backward, epoch checkpoint/resume를 확인한다. production 설정은 바꾸지 않는다.

**6단계 — 연구 결과 종결:** 두 arm의 source/event/jitter/학습·데이터 정의를 대조하고, 새 feedback checkpoint와 예측 출처를 연결한 paired evaluation/statistics를 만든다. feedback 독립 효과를 말하려면 matched-quality control 등 별도 사전 정의가 필요하다.

## 15. 이번 작업에서 하지 않은 것

원본 코드 수정, model/graph/data/epoch 축소, 원격 Git pull/push, 사용자 서버 접속/프로세스 종료, trainer 덮어쓰기, medical 데이터 이동/삭제, 기존 result/journal/manifest 수정, 전체 training/evaluation은 하지 않았다. source 159개는 감사 종료 시 다시 hash를 비교해 변경 없음이 확인됐다.

## 16. 파일별 대장과 원시 증거

- `HierCP_review_inventory_20260912.md`: 159개 파일을 빠짐없이 열거한 역할·검사 대장.
- `artifacts/review_ledger.json`, `review_ledger.csv`: file-local/code.txt 시작 줄·SHA·함수/클래스·실행 상태.
- `artifacts/static_audit.json`, `internal_imports.json`, `syntax_errors.json`: 전수 정적 검사.
- `artifacts/trainer_hash_audit.json`, `preservation_audit.json`: trainer/source 보존 hash 증거.
- `audit_probes.py`, `artifacts/audit_probes.json`, `audit_probes.log`: 추가 진단과 결과.
- `artifacts/tests_result.json`, `unittest.log`: 기존 회귀 실행 시도 전체 ID/오류/skip.

**최종 승인 상태: 핵심 구현 연결 확인, 일부 수치/운영 진단 확인. 위 결함·구조적 위험·실환경 미검증 때문에 전체 파이프라인 승인 보류.**


## 작업 완료 체크리스트

- [x] 원본 업로드와 159개 복원 소스가 보존됐는지 hash로 확인했다.
- [x] 전체 파일 인벤토리·Python/JSON 구문·정적 내부 참조를 검사했다.
- [x] 모델/graph/후보/해상도/epoch/physical batch 설정을 감사 편의로 바꾸지 않았다.
- [x] raw transport·streaming autograd·hierarchy·feedback·복구·평가의 주요 경계를 별도로 대조했다.
- [x] synthetic DEBUG와 실제 의료/GPU 실행을 구분하고 경계 double을 표시했다.
- [x] local CPU 환경과 의존성 부족, 기존 suite 실행 실패를 기록했다.
- [x] 서버나 사용자 원격 세션/실험 결과를 변경하거나 종료하지 않았다.
- [ ] PyG 전체 hierarchy의 native forward/loss/backward를 이번 환경에서 재검증했다.
- [ ] 실제 nnU-Net complete epoch 및 checkpoint resume를 검증했다.
- [ ] 전체 의료 cohort의 bank/40-epoch GNN/250-epoch 두 arm을 실행했다.
- [ ] 새 feedback 결과의 paired evaluation·통계·checkpoint provenance를 완료했다.
