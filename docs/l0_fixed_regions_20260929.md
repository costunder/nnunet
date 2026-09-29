# v2.2 고정 영역 계층 + GraphSAGE 구현·단기 검증

## 결정과 기존 checkpoint

새 분할용 CNN을 먼저 만들어야 한다는 조건을 추가하지 않는다. **기존 학습 checkpoint의 CNN을 고정 snapshot으로 읽고**, downstream CNN은 별도 복제해 계속 학습한다. 사용자가 이미 전달한 서버 기록은 `/home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_gnn40_resume114_20260928/training/checkpoint_latest.pt`, 마지막 전달 step169다. 이 로컬 작업에서 서버 파일을 읽거나 최신 step을 확인한 것은 아니다.

로컬 연결·비용 검사는 실제 CT DEBUG 학습의 `work/v22_rereview_20260927_DEBUG/uninterrupted/checkpoint_latest.pt`를 읽었다. 저장 상태는 step4 / epoch1 / final_memory / debug=true다. CNN24개 state 항목을 strict load하고 checkpoint 파일 및 CNN 내용 SHA를 기록했다. **무작위 CNN 대체가 없으며, step4를 충분히 학습된 분할 모델이라고도 부르지 않는다.** 기존 checkpoint의 존재와 영역 분할의 타당성은 다른 검증 항목이다.

## 새 구조

별도 `l0_regions` 패키지를 추가했다. 기존 fine SAGE·동적 EZ-SP·Basic CP 코드는 그대로 보존했다.

```text
준비 한 번: 기존 CNN snapshot(eval, frozen)
  → 고정 기준 view의 CNN32 특징
  → 공식 EZ-SP 1차 partition / typed quotient
  → 원래 node mass로 모은 고정 CNN32 특징
  → 공식 EZ-SP 2차 partition / typed quotient
  → record별 할당표·두 scale 관계 edge·mass·shell·좌표·원본 CT 저장

매 update: 현재 학습되는 CNN
  → 원래 표본 위치들의 최신 CNN32 특징
  → 고정 1차 할당표로 mean 집계 →128D projection
  → 작은 그래프 SAGE 2층
  → 고정 2차 할당표로 original-mass weighted mean
  → 작은 그래프 SAGE 1층
  → 두 scale mass-aware readout → 기존128D fuser → 기존L1/L2/loss
```

2차 partition도 사전에 고정 CNN32에서 생성한다. 학습 중 SAGE128로 재분할하는 이전 동적 모델과 다르다. 분할 결정에는 gradient가 없지만, live feature 집계에는 gradient가 있어 CNN까지 전달된다. Fine graph와 수학적으로 동일한 모델이라는 주장은 하지 않는다.

13종 관계는 원래 edge의 영역 간 quotient만 사용한다. 같은 관계의 중복 영역 쌍은 합치고 내부 same-role edge는 제거한다. 새 radius/kNN/완전연결, seed384/hop 재확장은 없다. SAGE는 **unique coarse neighbor의 평균**을 사용하므로 원래 fine neighbor mean과 동일하지 않다. Mass는 집계/readout에 쓰고 새 learned geometry 입력으로 붙이지 않는다.

## 캐시 재사용과 admission

- 원본 cache SHA, record SHA 및 shared source, donor/component, 후보 중심, frozen CNN SHA, 두 reg와 pinned 공식 구현, 기준 view epoch/index, stride4, 준비 소스 SHA에 결속한다. 원본 record SHA는 저장된 transform·geometry를 함께 결속한다.
- 기준 view를 명시적으로 고정한다. Epoch별 새 fine view 위에 예전 할당표를 얹지 않는다. 서로 다른 view/model/profile은 collate에서 거부한다.
- 캐시 파일 SHA와 기대 identity, mass/coverage/좌표/역할/shell/edge index를 검사한다. GPU forward에서도 CT·coverage·assignment·cross-pair edge를 검사한다.
- 준비 단계에서 연결성, bbox·분산·상한을 기존 official adapter와 동일하게 검사한다. bbox·분산·상한은 미검증 초기 profile이며 공식 merge 함수의 내부 보장으로 취급하지 않는다.
- `--allow-unvalidated-profile`은 자원 제한이 있는 **diagnostic-only** 계산 허용이다. 위반은 보고서에 남고 production admission은 false다. 이 옵션 없이 위반 partition을 사용하면 오류다. 모델의 production checkpoint export도 거부한다.
- 기존 두 reg **0.02/0.02**, min_size1, 초기 상한을 단기 비교에서 그대로 사용했다. 자동 변경·node drop·record skip·fallback·CP no-op은 없다. Reg는 CLI 필수값이며 기본값이 없다.
- 캐시에는 fine sampling 좌표는 있지만 **fine edge 목록은 없다**. Training RegionBatch가 전송하는 edge는 두 coarse scale뿐이다. 비교 process에는 baseline 측정을 위해 별도 fine batch도 상주하므로 process 전체에 fine edge가 없다고 주장하지 않는다.
- Independent record 파일 읽기·쓰기는8개 worker thread, 모델 입력은8pair disjoint batch. 같은 source CT는 collate에서 deduplicate한다. Record별 캐시를 다른 physical batch로 다시 묶는 검사를 통과했다.

## 실제 그래프 크기와 비용

실제 DEBUG inner_train8개 전체, 동일 기준 view epoch0/index0, 동일 trained-CNN snapshot, FP32, query physical8/effective8/accumulation1, CNN chunk4. RTX5070Ti16GB, workers8, CUDA allocator6GiB/RSS12GiB/240초의 명시적 DEBUG 예산. 장기 학습이 아니다.

| 단계 | 메시지 전달 node 합계 | directed edge 합계 |
|---|---:|---:|
| 기존 fine SAGE | 52,666 | 2,761,602 |
| 고정 1차 영역 / SAGE2 | 13,868 | 379,049 |
| 고정 2차 영역 / SAGE1 | 13,849 | 378,284 |

| Pair | 1차 nodes | 1차 edges |
|---|---:|---:|
| liver_66:19 | 1,464 | 35,047 |
| liver_66:1 | 691 | 17,414 |
| liver_75:44 | 3,856 | 97,107 |
| liver_75:2 | 978 | 32,839 |
| liver_72:84 | 903 | 33,128 |
| liver_72:1 | 1,277 | 34,709 |
| liver_71:91 | 2,969 | 79,604 |
| liver_71:0 | 1,730 | 49,201 |

**2차는1차에서19 nodes만 더 줄었다.** 이 설정으로 충분한 두 단계 계층을 얻었다고 볼 수 없다. 현재 bbox·분산·상한 위반도 그대로 남아 `profile_exceeded=true`다. 압축률 또는 속도를 이유로 적합한 partition이라고 통과시키지 않았다. 두 scale의 role/shell cluster·연결성분·mass 분포·초과 비율·공식 병합 시간은 원시 `partition_audit`에 있다. Artifact admission flag는 준비 batch 전체의 보수적 결과이고, pair별 타당성 판정은 audit로 구분해야 한다.

아래는 **같은 측정 실행 안의** 워밍업1회 이후 교대3회 평균이다. 이전 GAT2.283초/SAGE1.203초와 섞어서 배속을 계산하지 않는다.

| 항목 | fine SAGE | 고정 영역 SAGE |
|---|---:|---:|
| update | 1.3760초 | 0.6140초 |
| forward | 0.6066초 | 0.2473초 |
| backward | 0.7235초 | 0.3195초 |
| 별도 support embedding refresh,8records | 0.4585초 | 0.1922초 |
| peak allocated | 약0.791GiB | 약0.342GiB |

두 모델의 대응 parameter가 동일한지 검사한 뒤 각 trial을 독립 복제했다. CNN snapshot은 동일하고, 새 SAGE 및 공통 L1/L2 등의 나머지 가중치는 seed42 초기값이다. 기존 GAT optimizer/memory를 resume한 실험이 아니다. 기존 observed-rank/aux/alignment loss로 forward→backward→optimizer를 실행했고 CNN·SAGE 각층·L1/L2의 nonzero finite gradient를 검사했다. Model master hash는 변하지 않았다.

동일 checkpoint의 사전 순차 I/O 진단은1.230→0.619초였다. 최종 병렬 I/O 진단과 두 원시 결과를 모두 보존했다. 짧은 반복의 편차를 숨기지 않는다.

## 준비·용량까지 포함한 해석

최종 실행에서 partition 준비3.307초, 파일 저장1.157초, 최초 읽기/검사/collate4.866초였다. Fine loader/H2D와 region H2D까지 합친 **준비 및 최초 사용 비용은9.890초**다. 8pair 캐시 파일과 manifest 합계 **23,947,998bytes(23.95MB /22.84MiB)**다. 전체14,102개 캐시는 생성하지 않았다.

같은8pair에 대한 update 절감은0.762초다. 이 차이가 계속 유지되고 최초 준비 비용을 한 번만 낸다는 가정이면 약13회 재사용해야9.890초를 회수한다. 이는 local DEBUG 조건의 산술이며 production epoch 예상이 아니다. Cache를 매번 디스크에서 다시 읽고 검사하면4.866초가 반복될 수 있어 이득이 없어질 수 있다. Warm RAM cache의 production loader나 전체 cohort의 저장 비용까지 검증됐다고 주장하지 않는다.

단순 평균으로 전체 저장량을 외삽하거나 현재8개를 전체 graph 크기 분포의 대표라고 보장하지 않는다. 새로운 후보/transform/view/checkpoint에는 새 준비 비용이 필요하다. 서버MIG/batch32/full support/BF16/validation/checkpoint 시간은 이번 비교에 포함하지 않았다.

## 구현·검증 상태

- `l0_regions/preparation.py`: pinned official partition, frozen two-scale hierarchy, typed quotient, per-record artifacts와 binding.
- `l0_regions/data.py`: 무결성 검사, 새 파일 저장, identity/hash load, fine edge 없는 rebatching/transport.
- `l0_regions/encoder.py`: live CNN pooling + SAGE2/1 + two-scale readout, diagnostic admission 및 자원 확인.
- `tools/compare_fixed_regions_debug.py`: 기존 checkpoint에서 CNN만 읽어 같은 실제8pair 비교. 원본 checkpoint가 읽는 동안 바뀌면 오류. 원본 쓰기/자동 장기 학습 없음.

기존23개 + 신규6개 = **29개 검사 통과**, skip0. 새 검사는 두 단계 typed quotient의 원본 edge 대응, 파일 변조/다른view·CNN·소스 거부, 잘못된 mass/parent/grid/edge 거부, batch/단독 동등성, online partition 금지 상태에서 실제 CNN gradient/optimizer update, 실패 profile의 일반 실행 거부다.

초기28개 검사 중1개는 sandbox의 임시 파일 권한 오류였고, 프로젝트 work 아래 새 fixture를 보존하는 경로로 실행해 통과했다. 실패 사실은 verification에 남겼다. 최종29개 이후 stale inherited profile ID/설명을 fixed-region 모델 ID로 정정하고 관련6개를 재검사했다. 수치/merge 옵션은 바뀌지 않았다. 이전 측정의 소스 hash와 cache binding은 덮어쓰거나 새 identity로 재표시하지 않았다.

[최종 비용 보고서](../validation/l0_fixed_regions_20260929/report.json) · [검사 기록](../validation/l0_fixed_regions_20260929/tests.txt) · [검증 요약](../validation/l0_fixed_regions_20260929/verification.json).

실행 가능한 새 diagnostic 경로까지 연결했지만, **production cache builder/loader·전체 학습 runner로 전환한 상태는 아니다.** 영역 타당성, 효과적인2차 partition, 전체 cohort 메모리/스토리지 계획, CP 추천/segmentation 효용이 남아 있다. Basic CP/L1/L2/loss/후보128/원본 mask는 변경하지 않았고 전체 학습·평가를 시작하지 않았다.

## References

- [공식 Superpoint Transformer / EZ-SP 저장소](https://github.com/drprojects/superpoint_transformer): 작은 convolutional embedding을 학습하고 인접 embedding으로 영역을 만드는 접근을 확인했다. 이를 우리CT에서 검증된 embedding이라고 일반화하지 않는다.
- [EZ-SP 논문](https://arxiv.org/html/2512.00385v2).
- Official merge: `drprojects/torch-graph-components`, commit `e3db9f352fae52dff416616742b3c7ff1378451d`. 기존 vendored 함수를 변경하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.3층/128D 유지.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.사용자가 지정한 영역 quotient로 표현 단위를 변경, 원래 표본 mass 보존.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.명시적DEBUG8개 전체, cap으로 제거 없음.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.동일batch8, I/O8workers.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.이번 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.합성 단위검사와 실제CT/기존checkpoint 비용 검사 구분.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
