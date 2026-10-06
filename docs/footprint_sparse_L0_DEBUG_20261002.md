# v2.2 L0: 완전한 donor footprint 기반 국소 그래프 — 실제 CT DEBUG

2026-10-02 최종 수치 근거는 `work/footprint_sparse_CT_DEBUG_20261002_r2/report.json`이다. CNN 계산용 직육면체 crop와 그래프의 노드 선정 영역을 분리했다. 시각화 테두리만 없앤 변경이 아니다. 다만 **원본 CT의 5 mm Z 간격과 격자 표본은 그대로이며, 그래프의 추천 정확도·최적 크기·production 준비 완료를 검증한 결과는 아니다.**

## 변경 범위와 구현

비교 코드에서 `baseline`은 기존 `RelayedV1SparseL0`의 직육면체 crop 영역 + native6 축 방향 경로이고, `relational`은 새 `FootprintPhysicalSparseL0`의 완전한 donor footprint 영역 + conservative26 물리 경로다. 두 모델의 trainable architecture, 초기 가중치와 원본 query는 같다. 새로운 graph-domain 규칙과 path substrate는 명시적인 DEBUG 비교 설정이다.

```text
변경하지 않은 원본 간 내부 CT crop, native spacing
    → 같은 8-convolution CNN [12, 24, 32]
    → 같은 전체 native 위치별 CNN 특징 풀
    → 새 graph-domain: 완전한 donor footprint에서 물리 거리 ≤ 기존 margin 10 mm
                         ∩ 원본 간 mask ∩ 모든 CNN scale의 유효 표본
    → 현재 CNN 특징 + 공간 정보로 shell별 FPS seed 선정
    → conservative26/supercover가 확인된 물리 경로의 relay 보존
    → 기존 8종 관계 + 동일 반경 안의 bidirectional mandatory parent 관계
    → 기존 3-layer 128D 관계별 mean SAGE + 역할/shell attention readout
    → 기존 donor–recipient fusion → 128D → 변경하지 않은 L1/L2/loss
```

Near/mid/wide는 원본 anchor로부터 각각 ≤5 mm, 5–10 mm, >10 mm다. shell당 16/32/64 seed, 즉 **branch당 48/96/192 seed**를 각각 비교했다. 이 숫자는 relay를 포함한 최종 노드 상한이 아니다. 기존 관계의 반경 6/8/5/8 mm와 incoming nearest-source 수 3/3/1/3을 유지했다. 추가 parent 관계는 동일 6 mm 이내 실제 누적 경로 길이를 검사한 후 합쳐지므로 최종 incoming degree는 nearest-source 수를 넘을 수 있다.

`footprint_data.py`는 기존 `sources`에서 선택한 실제 donor component의 **전체 full_mask**를 읽는다. Donor의 모든 occupied native voxel을 원본 anchor에 대한 상대 mm로 보존한다. Recipient에서는 공식 `hiercp_v22.data.donor_in_target_spacing`으로 같은 source를 recipient native spacing에 변환한 전체 patch mask와 명시적 anchor를 쓴다. Recipient의 종양 GT로 그래프 영역을 만들지 않는다. Footprint 내부를 제거하는 설정도 추가하지 않았다.

`footprint_domain.py`의 영역은 모든 footprint voxel centre에 대한 최소 Euclidean 거리 ≤10 mm다. Donor bbox의 각 축을 10 mm씩 늘린 상자가 영역의 정의가 아니다. 원본 CNN crop는 계산 입력으로 그대로 남으며 새 영역은 그 crop의 원래 유효 특징 풀 안에서 검사된다. 따라서 crop 밖 CT를 새로 인코딩하거나 연속 공간 정보를 복원한 것은 아니다.

`physical_relay.py`는 실제 유효 native voxel에서 26방향의 양의 물리 길이 경로를 계산한다. 대각선은 supercover에 해당하는 모든 native 표본이 유효해야 하므로 두 endpoint만으로 background corner를 건너뛰지 못한다. 압축 parent 경로의 누적 길이는 기존 6 mm 제한을 넘지 않는다. 이 검사는 **추가 mandatory 경로**의 증명이다. 기존 radius/nearest 관계까지 혈관이나 해부학적 통로라고 주장하지 않는다.

## 전체 mask와 입력 결속

실제 고정 donor는 `liver_1`, component 1이며 recipient는 `liver_66`이다. 원본 donor full mask는 **769 voxel**이다. Recipient spacing의 공식 변환 mask는 **737 voxel**, patch shape `[33,25,7]`, 명시적 anchor `[16,12,3]`이다. 원본 source spacing은 `[0.6757810116,0.6757810116,5.0]` mm다.

각 scene에 원본/변환 mask hash, 전체 voxel 수, 원본 anchor, spacing, record ID, P/U, donor assignment, crop 범위와 CT/crop/간 밖 voxel 수를 기록했다. Tensor version 및 metadata signature가 바뀌면 거부한다. CPU source 전처리와 compact patch geometry cache를 사용하며, source의 전체 CT image/full-volume mask를 새 cache에 복제해 보존하지 않는다. 완전한 mask의 hash와 geometry는 보존한다. Compact geometry 8개 cache entry의 최종 byte 합계는 **205,888 bytes**이며 원본 raw/crop와 함께 resident budget에 계상한다.

133개의 donor scene와 133개의 recipient scene 모두 전체 footprint를 기록했다. CT/crop 밖 footprint voxel은 모두 0이다. **Recipient 46개 record에서는 간 밖 footprint voxel이 있으며 최대 475 voxel**이다. 이를 clip하거나 그 record를 버리거나 P/U를 변경하지 않았다. 관측 위치와 특정 foreign donor의 실제 paste 가능 여부는 다르므로 이 값은 그대로 진단 기록으로 남긴다. Online CP의 기존 **전체 paste mask admission**은 유지해야 한다. 이번 DEBUG는 해당 조건을 통과한 CP 결과나 segmentation 학습 완료가 아니다.

정답 계약은 **P = 실제 CT에서 관측된 적격 종양 anchor**, **U = 종양이 관측되지 않은 기존 비교 위치**다. P를 특정 donor의 CP 적합 정답으로, U를 CP 부적합 정답으로 다시 정의하지 않는다. Donor/spacing/footprint는 조건 입력과 기하 결속이며 정답을 바꾸는 근거가 아니다.

별도의 실제 P5/U27 physical32 비교에서 기존 `CropStore`와 새 reader의 `images`, `organ`, `donor`, `recipient`, `indices`가 **GPU bitwise equal**이고 native crop audit도 정확히 같았다. 입력은 `[33,1,58,50,7]`이며 33은 공유된 donor crop 1개 + recipient crop 32개의 수다. 이 전처리 대조 5.215740초는 update 측정 밖이다.

## 실행 규모와 가중치

- 실제 `liver_66`의 **P5 + 기존 U128 = 133 record 전부**를 두 sampler와 세 seed 규모에서 forward했다. 고정 physical32의 마지막 batch만 자연스럽게 5개다.
- 원래 DEBUG loss context 139개 관측, 전체 schedule 8 tile·643 P×U 비교를 보존했다. 수치 학습 검사는 그 schedule에서 원본 P5/U27 tile 하나를 사용했다. 133개 전체의 학습 epoch를 완료한 것은 아니다.
- 여섯 비교 모델마다 disposable warmup 1회 뒤 가중치·buffer·RNG를 복원하고 fresh AdamW로 연속 update 3회, **총 18회 측정 update**를 실행했다. Optimizer state step은 모두 3이다.
- CNN, node projection, role embedding, SAGE, readout, attention pool, fusion, L1, L2의 실제 loss gradient와 optimizer 변화가 여섯 branch의 모든 측정 update에서 확인됐다.
- 원래 DEBUG memory 8개를 모델별로 전부 재인코딩하고 기존 patient-group exclusion 후 support와 teacher를 각각 다시 만들었다. 세 update 동안 각 branch의 자체 support/teacher는 고정하고, 종료 후 갱신 비용을 별도로 측정했다.
- CNN/fusion/L1/L2는 무결성이 확인된 **로컬 DEBUG step4** checkpoint에서 복사했다. Graph module은 새 seed42 가중치다. 서버에서 장기 학습한 모델을 사용하거나 옛 checkpoint를 exact resume한 결과가 아니다.
- 전체 모델 parameter는 모든 branch에서 **1,785,558**이며 공통 초기 model hash는 `0ebc556683ae799cd65f722441ea8387b6539afadc7192bc05031063a7ad75a5`다. CNN/L1/L2 깊이·너비, loss, Basic CP, 후보128, 원본 mask, native resolution을 함께 변경하지 않았다.

장비는 RTX 5070 Ti 한 개, VRAM 약 15.92 GiB, CPU 8 physical/16 logical, RAM 약 63.93 GiB다. DEBUG 한도는 CUDA12/RSS32/resident12 GiB, reader4, pinned memory와 nonblocking 전송이다. FP32, autocast/TF32 off, deterministic 모드다. 여섯 비교 모델을 함께 GPU에 올린 프로세스의 peak이며 단일 production 모델의 최대 입력 메모리 상한이 아니다. 최종 RSS는 약 6.04 GiB, coverage 포함 GPU peak는 약 4.75 GiB였다.

## 실제 형태와 연결 진단

아래 N/E는 전체 133개 record에서 **donor+recipient 한 pair**의 최솟값–최댓값이며 E는 directed typed edge 수다.

| Branch당 seed | 기존 N / E | 새 N / E | 새 weak component / isolate 범위 |
|---|---|---|---|
| 48 | 256–300 / 1,183–1,365 | 207–242 / 1,046–1,236 | 1–2 / 0–1 |
| 96 | 439–524 / 2,254–2,581 | 368–439 / 1,957–2,348 | 1–2 / 0–1 |
| 192 | 741–902 / 4,028–4,746 | 631–760 / 3,499–4,247 | 1 / 0 |

기존 native6 comparator는 133개 모두 weak component 1, isolate 0이었다. 새 sampling domain은 recipient별 원래 유효 풀 5,774–18,816개 중 3,103–8,775개를 포함했다. 기록별 비율을 평균하면 약 50.48%다. 이는 전체 fine CNN 특징 풀을 삭제한 비율이나 암 관련 정보 손실률이 아니라 **정의한 물리 context 영역 안의 후보 voxel 비율**이다.

현재 표시한 `liver_66:10`, branch seed96에서 실제 crop face 위 recipient context node는 **63/265 → 5/208**, donor는 **39/235 → 2/197**로 줄었다. 좌표를 화면에서 비틀거나 임의의 곡선으로 옮긴 결과가 아니다. 간과 CT의 좌표는 그대로다. 그래도 5 mm Z sampling의 층과 native 격자성이 남으며, crop face count가 0이 되는 조건을 만들지는 않았다.

**`liver_66:65`의 recipient seed 1개는 새 physical path substrate에서 세 규모 모두 root에 도달하지 못했다.** 해당 seed와 record는 유지·보고했다. seed48/96에서는 실제 joint graph도 component2/isolate1이고 seed192에서는 기존 typed 관계를 합친 joint graph가 component1/isolate0이지만 **physical forest의 unreachable1은 그대로**다. Weak graph 연결성과 supercover physical reachability를 같은 의미로 취급하면 안 된다. 이 record를 정상 production sample로 승인하거나 숨긴 것이 아니며, 전체 long training/production-ready 표시는 없다. 실제 세 update의 P5/U27 tile에는 이 record가 포함되어 있지 않다.

## Coverage는 같은 원본 풀에 대해 평가

새 영역의 바깥 box-corner 표본도 평가 기준에서 삭제하지 않았다. Recipient별 원래 전체 CNN 특징 풀, 합계 **1,938,845개 표본**을 동일하게 평가했다. 이는 crop들의 합이며 간 전체의 독립 voxel 수가 아니다.

| Branch당 seed | 원본 전체 풀 spatial 평균 mm, 기존 → 새 | CNN feature deficit 평균, 기존 → 새 |
|---|---|---|
| 48 | 5.7743 → 6.6127 | 0.07385 → 0.08765 |
| 96 | 4.4440 → 5.3833 | 0.05478 → 0.07294 |
| 192 | 3.4195 → 4.4069 | 0.03994 → 0.06208 |

**같은 원본 전체 풀의 coverage 수치는 새 영역에서 악화됐다.** Context domain을 바꾼 만큼 이전 seed를 전부 보존하거나 coverage가 개선된다는 단정을 하지 않는다. 이 수치는 spatial/CNN landmark coverage이며 추천 정확도나 유효 특징 보존률을 증명하지 않는다.

## 시간과 캐시의 정확한 측정 경계

첫 수치 probe r1을 보존한 후 r2에서는 한 immutable batch의 **완전한 footprint 거리/domain 및 native physical forest만** cache한다. Batch signature, anchors, spacing, organ/all-scale support, complete mask가 같을 때만 재사용한다. Feature가 바뀌는 CNN 결과·FPS seed·retained message graph·L0 출력·support teacher는 cache하지 않는다. 단일 batch geometry는 **46,045,196 bytes, 약 43.91 MiB**다. Warmup receipt는 domain/forest reused 둘 다 false, 새 branch의 세 측정 update는 둘 다 true였다.

| Branch당 seed | 기존 update 중앙값 초 | 새 cached update 중앙값 초 | 새 최초 warmup 전체 update 초 | 기존 → 새 post-update memory/teacher refresh 초 |
|---|---|---|---|---|
| 48 | 0.528091 | 0.512681 | 1.804533 | 0.370267 → 1.317194 |
| 96 | 0.697821 | 0.617820 | 1.837597 | 0.487246 → 1.443165 |
| 192 | 1.048026 | 1.002184 | 2.394699 | 0.692474 → 1.732079 |

Cached update는 이 동일 tile에서 약 2.9%/11.5%/4.4% 줄었다. **큰 epoch 시간 단축을 확인한 결과가 아니다.** 새 batch에서는 최초 domain 구축 비용이 남고 전체 memory refresh는 오히려 느리다. 새 최초 warmup의 GPU domain 계산은 1.142/1.113/1.255초이며 physical26 relay event는 0.129/0.116/0.123초다. 후자는 forest 생성과 retained-node 처리를 합친 event이므로 **forest 단독 시간은 미측정**이다.

전체 133개의 여섯 모델 first-build forward·coverage·CPU export를 합친 구간은 **65.234767초**다. 각 batch의 최초 domain/forest 비용을 포함한다. Same prebound tile의 update는 loading, diagnostic delta CPU copy, production checkpoint 저장을 제외하며 CNN/FPS/retained graph/L1/L2/loss/backward/finite/clip/AdamW를 포함한다. 매 update 현재 CNN 특징으로 FPS를 다시 실행한다. 예를 들어 새 seed192의 평균 FPS GPU event는 약 0.493초로 여전히 남는다. Memory/teacher refresh는 새 입력 batch, 전체 기존 DEBUG support와 teacher 재구성 및 loading/transfer를 포함한다. 이 범위들을 섞어서 서버의 3시간 epoch가 해결됐다고 환산하지 않는다.

## 검사와 시각화

- 구현·구문 검사 완료. 최종 회귀 검사 **59 PASS: CUDA58 + metadata1**, 10.520초. CPU metadata 검사는 공식 GPU path 검증으로 대체 표기하지 않았다.
- 실제 CT P5/U128의 모든 forward, 같은 physical32 입력 parity, 여섯 branch의 실제 loss/backward/optimizer 및 연속 AdamW3 update 완료.
- 브라우저 **93개 검사 PASS**, JavaScript 오류0. 17개의 candidate/branch 관측을 원시 report의 실제 N/E/component/isolate 수와 대조했다. 전체 donor/recipient/관계 필터, 카메라, 간 전체 view, 모바일 화면을 확인했다.
- Renderer의 lossless roundtrip은 표시용 실제 node23,962개·ordered typed edge124,895개·native parent path42,984개와 GT 좌표를 보존했다. 이 값은 여러 candidate/규모/모델을 합친 표시 데이터 수이며 단일 graph 크기가 아니다.
- 시각화는 원본 P5/U3의 두 모델·세 seed 규모를 선택해 비교한다. Whole-liver view는 주변 해부 위치 확인용이며 간 전체를 모델이 인코딩했다는 뜻이 아니다. `liver_66:65` 실패 진단은 전체 수치 report에 남아 있다.
- 전체 GNN/nnU-Net 학습, 전체 cohort 정확도 평가, online CP/segmentation 성능 평가 및 production-ready 선언은 **하지 않았다**.

## 근거 파일

- [최종 실제 CT r2 report](<D:/AI project/nnunet/work/footprint_sparse_CT_DEBUG_20261002_r2/report.json>): SHA256 `678a767f6f11228032c49a7725520e86ce688c2d033481389c1dc9407cc50042`.
- [원본 visual payload](<D:/AI project/nnunet/work/footprint_sparse_CT_DEBUG_20261002_r2/visual_payload.json>): SHA256 `999aa7981113e3317fc57a946fef7c430456a187c166cf7e1ed1e3c3ab1d0488`.
- [브라우저 검사](<D:/AI project/nnunet/work/footprint_shape_20261002/visual_qa_r2/browser_checks.json>)와 같은 디렉터리의 new/old/whole-liver/mobile PNG.
- [최종 완료·보존 검증 기록](<D:/AI project/nnunet/work/footprint_shape_20261002/completion_receipt_r2.json>): 회귀59, 실제 CT r2, 브라우저93 및 source/기존 입력/결과 보존 집계.
- 실제 3D 표시 원본: `C:/Users/user/.codex/visualizations/2026/09/17/01a0ae06-ab24-7fe0-bdb8-ef2751a1c7be/footprint-context-ct-r2.html`.
- [검증 도구](<D:/AI project/nnunet/tools/verify_footprint_sparse_ct_debug.py>), [full-footprint reader](<D:/AI project/nnunet/l0_sparse_feature/footprint_data.py>), [physical domain](<D:/AI project/nnunet/l0_sparse_feature/footprint_domain.py>), [path substrate](<D:/AI project/nnunet/l0_sparse_feature/physical_relay.py>), [새 L0](<D:/AI project/nnunet/l0_sparse_feature/footprint_graph.py>).

기존 r1 및 earlier relay 결과·checkpoint·원본 assignment는 보존했다. r2 report에는 실행 source16개의 hash, source/assignment/checkpoint 불변 확인, mask content/binding hash가 함께 들어 있다. 가중치 checkpoint SHA256은 `9b36e85a57819042026b152e7bcc7236ed9ad0b372fa32b5d5c321d74b50c1f6`다. 이 문서에서 신규 전체 학습 실행을 요청하거나 자동 시작하지 않는다.

[독립 source 대조](<D:/AI project/nnunet/work/footprint_shape_20261002/independent_source_provenance_audit_r2.json>)에서는 r2 구현16개, production manifest121개, 기존 입력·결과5개와 기존 dirty 파일3개의 hash가 모두 일치했다. 불일치는 0개다. 이 CPU JSON/hash 검사는 GPU 수치 검사나 정확도 평가를 대신하지 않는다.

## 짧은 실제 CT DEBUG 재현

아래는 이번 수치 검사의 재현 경로다. 한 CT의 모든133위치 forward와 같은32pair의 짧은 복제 update만 실행한다. 전체 학습 명령이 아니다. 각 재실행은 존재하지 않는 새 output 경로를 사용해야 한다.

```powershell
& 'D:/AI project/nnunet/.venv/Scripts/python.exe' -u 'D:/AI project/nnunet/tools/verify_footprint_sparse_ct_debug.py' `
  --run 'D:/AI project/nnunet/work/local_cnn_experiment_resume_DEBUG_20261001/resumed' `
  --assignment 'D:/AI project/nnunet/work/v222_v1_full_training_20260924/cache/pair_assignment.json' `
  --assignment-receipt 'D:/AI project/nnunet/validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json' `
  --case liver_66 --output 'D:/AI project/nnunet/work/footprint_sparse_CT_DEBUG_20261002_rerun' `
  --query-radius-mm 3 --near-radius-mm 5 --mid-radius-mm 10 `
  --workers 4 --cuda-gib 12 --rss-gib 32 --resident-gib 12 `
  --coverage-workspace-mib 96 --warmup-updates 1 --measured-updates 3
```

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 노드 domain 변경은 box-shape 문제에 대한 명시적 DEBUG 비교이며 133개 관측·동일 seed 규모를 모두 보고했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. DEBUG와 표시 P5/U3의 범위를 전체 forward133과 구분했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 physical32·disjoint paired graph·네 개 parallel reader를 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 최종 DEBUG에서 OOM은 발생하지 않았고 geometry 재계산·cache byte·fresh/steady 비용을 분리했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 새 graph parameter의 seed42 초기화를 학습된 추천 가중치로 주장하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 확인 범위는 여섯 모델·실제 동일 tile의 18개 DEBUG update다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. Unreachable seed, 악화된 coverage, 느린 first-build/refresh도 기록했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
