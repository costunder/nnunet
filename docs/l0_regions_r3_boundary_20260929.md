# v2.2 R3 검토 반영 — RAM 검증 경계와 배치 전환 비용

2026-09-29. 같은 v2.2의 버그 수정/측정 보완이다. 새 모델 버전으로 승격하지 않는다. 기존 F01/F02/F03, adapter와 모든 과거 결과를 보존했다.

## 반영한 지적

- **R3-B01:** `check_verified`가 표시 없음에도 False만 반환하던 동작을 ValueError로 바꿨다. Resident cache hit, RegionBatch.to, RegionSAGEEncoder.forward 모두 표시 존재·일치를 요구한다. Encoder의 구조 검사 대체 경로도 제거했다. 현재 내용으로 자동 재인증하지 않는다. 최초 인증은 기존 receipt/내용 검사를 거친 load/collate, 정상 전송은 검증된 원본에서 파생하는 경로이다. 매 update 파일 hash나 새 모델 hash 체계를 추가하지 않았다.
- **P3:** 비교 도구의 `cache_tensor_bytes`를 `cache_serialized_pt_bytes`로 정정하고, `resident_tensor_storage_bytes`, process RSS snapshot, pinned memory 및 일시적 메모리 측정 범위를 별도로 기록한다. 내부 tensor 예산 방식은 원래 올바르므로 변경하지 않았다.
- **측정 경계:** warm GPU batch/CSR update와 연속 batch 비용을 분리했다. `tools/profile_region_schedule_debug.py`는 기존 `v1_training.groups`를 호출하며 전체 명시적 DEBUG 관측을 모두 처리한다. 순서와 실제 batch 크기를 원문에 기록한다. 조회→전송/검증→CSR→기존 forward/loss→backward→clip→optimizer를 동기화 계측한다. Offline 준비·support/plan은 별도, production 저장은 수행하지 않는다.
- Basic CP, L1/L2, rank/aux/alignment loss, 후보128, mask, reg0.02/0.02, min_size, 초기 admission 상한, 모델 깊이·너비를 변경하지 않았다. 단순히 속도나 PASS를 만들기 위해 profile을 조정하지 않았다.

## 검사

CPU/GPU 회귀 **37개 통과, skip 0**, 19.407초. 합성 topology 검사에는 공식 GPU partition과 실제 CNN/SAGE/L1/L2 gradient·AdamW 연결이 포함된다. 실제 CT 비용 측정과 구분한다.

추가 반례: 표시 삭제, 삭제 후 CT in-place 수정, binding metadata 변경, tensor 교체. Cache hit·CPU/CUDA 전송·encoder에서 모두 거부하며 CNN 호출 전 차단을 확인했다. 정상 cold/hit/CPU/GPU 전송과 forward, storage alias 중복계산 방지 및 파일 크기/실제 storage 차이를 함께 검사했다.

## 실제 CT DEBUG 결과

기존 8관측 전부, physical 설정8, workers8, FP32, RTX5070Ti. CNN은 기존 step4/epoch1 DEBUG checkpoint, 나머지는 seed42 초기값. GPU6GiB/RSS12GiB/240초의 명시적 한도. 시작 시 다른 GPU 작업이 있었다(9500MiB, 사용률77%); 다른 프로세스는 변경하지 않았다. 단독 장치 benchmark나 서버 MIG 측정이 아니다.

| 측정 경계 | Fine SAGE | Fixed-region SAGE |
|---|---:|---:|
| 동일 GPU batch8·warm CSR update, warmup 제외3회 | 1.529641초 | 0.836438초 |
| 연속 schedule, CPU RAM hit 후 step 평균 | 0.515162초 | 0.605477초 |
| 위 step 중 H2D·검증 | 0.004388초 | 0.100538초 |
| 위 step 중 CSR 구성 | 0.039629초 | 0.065739초 |
| 위 step 중 forward/loss | 0.209541초 | 0.181638초 |
| 위 step 중 backward | 0.237744초 | 0.230107초 |

**두 행은 batch cardinality가 다르다.** 기존 scheduler는 로컬4환자×2관측을 `[[6,7],[4,5],[3,2],[1,0]]`으로 묶는다. 따라서 연속 경로의 실제 physical/effective batch는 양쪽 모두2, accumulation1이다. 설정8을 낮추거나 tail을 버린 것이 아니며 모든8개 관측을 사용했다. 전체8개 동시 비교와 이 수치를 같은 작업량으로 비교하지 않는다. 서버 physical32/full-support 검증도 아니다.

연속 경로는 2회 전체 DEBUG schedule 순회다. 각 arm/pass는 독립 clone·새 AdamW로 시작하고, 해당 pass의4개 batch 사이에는 모델·optimizer 상태를 계속 유지한다. 순서를 교대했다. 첫 순회 region은 cold load/검증/collate 포함 평균2.835507초, 다음 순회 region RAM hit는0.605477초다. Fine CPU batch 최초 준비 비용은 별도 `fine_cold_load`에 기록했다. Cold region과 이미 준비된 fine을 동등한 cold 비교로 주장하지 않는다.

**RAM hit 후에도 이번 연속 step은 영역 모델이 약17.5% 느렸다.** CSR은 fine1회, region2회 매 새 GPU batch에서 구성됐다. 같은 warm batch에서의45.3% update 단축을 epoch 전체에 적용할 수 없다. Support/plan까지 더한 로컬4step 순회는 양쪽 약2.79초로 비슷하지만, 매우 짧은 공유 GPU 진단이므로 일반화하지 않는다. 새 전송/검증 경계를 우회해 수치를 좋게 만들지 않았다.

## 메모리 단위 정정

- 저장된 `.pt` 파일 합계: **25,925,496bytes**. R3 과거 보고의 같은 숫자는 RAM이 아니라 이 파일 크기였다.
- 8관측 한 batch의 실제 고유 tensor storage: **23,254,500bytes**.
- 연속4batch를 보관한 region tensor storage: **23,696,868bytes**. Batch 조합별 entry4개, miss4/hit4. Record 단위 중복제거 캐시로 주장하지 않는다.
- 첫 resident load 후 process RSS: **2,107,731,968bytes**. Python/runtime 등을 포함한 순간 snapshot이며 peak가 아니다.
- 이 진단은 pin_memory를 호출하지 않으므로 별도 pinned tensor storage는0. Miss의 임시 item/collate peak는 미측정. 예산은 retained tensor 기준이고 ResourceBudget의 RSS 검사는 단계 경계이다.

## 남은 문제와 근거

2차 축약이 미미한 문제, fine mass65.42%가 미검증 bbox/분산 초기 profile을 넘는 문제, 고정 CNN 분할 품질은 이번 수정으로 해결됐다고 표시하지 않는다. 65.42%는 검증된 임상 오류율이 아니다. GraphSAGE coarse unique-mean은 fine/GAT exact resume가 아니다. 전체 cohort/MIG epoch·validation/저장·CP 효용은 미검증이다.

원시 기록: `validation/l0_regions_r3_boundary_20260929/tests.txt`, `gpu_console.txt`, `report.json`, `measured_source.zip`. 생성 region artifact는 `work/l0_regions_r3_boundary_20260929_DEBUG/region_cache`에 보존한다. Source snapshot의 측정 전후 bytes 일치 확인. 이번 문서화의 새 수정은 모델 학습 소스와 구분한다. Production checkpoint/ready, 장기 GNN/nnU-Net 학습·전체평가·git push는 수행하지 않았다.

첨부 본문은 전부 읽었지만 본문의 sandbox 링크에 있는 별도 evidence ZIP/전체 MD는 로컬 첨부가 아니므로 읽었다고 주장하지 않는다. 반례는 현재 로컬 코드에서 직접 검사했다.

## 추가 r3 독립 검토 대조 — 74537553 첨부

이 첨부는 r4가 아니라 이전 r3/574파일/35검사 전달본을 검토했다. F01/F02/F03 수정 확인, P3 파일 크기 정정 및 D01/D02 비용 경계는 위 r4 수정·측정으로 대응된다. 새 P1/P2를 재현하지 못했다는 문장을, 앞선 별도 검토에서 재현하고 r4에서 막은 R3-B01 반례의 부정으로 해석하지 않는다.

- 기존 `groups`를 실제 로컬 DEBUG metadata에 적용해 epoch0~3을 읽기 전용으로 확인: 순서만 바뀌고 4개 query chunk의 내부 index 순서는 유지됐다. 새 GPU 학습 epoch를 실행한 것이 아니다. 동일 fixed view/binding일 때 RAM 재사용이 가능한 구조이며 다른 view·memory batch·후보 조합까지 같은 key라는 뜻은 아니다.
- 기존 실제 region artifact에서 CPU A→B→A는 동일 A 객체를 반환했다. 두 entry의 retained tensor storage는12,079,312bytes. 전체 query4개 key의23,696,868bytes는 앞선 r4 진단 수치이다. 전체 cohort 준비량으로 확대하지 않는다.
- RTX5070Ti에서 실제 두 scale CSR을 A→A→B→A 순서로 준비했다. scale별 누적 build는 각각 **1→1→2→3**. 동일 A 재사용0.000636초, B 후 A 복귀0.040605초. 워밍업/공유GPU 영향을 포함한 단회 동작 확인이며 새 속도 비교 평균으로 사용하지 않는다. CNN/update/학습 없음.
- `cache_total_bytes=25,966,126`은 sidecar 포함 파일 합계, `.pt`만25,925,496, 8pair resident tensor23,254,500으로 구분한다.
- 원시 추가 근거: `validation/l0_regions_r3_boundary_20260929/additional_r3_review_schedule_audit.json`. 기존 r4 ZIP/37검사/실제 update 결과를 덮어쓰지 않았다. 이번에는 모델·loader·cache 구현을 추가 변경하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 설정8·실제 tail2 구분, workers8.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 반례 검사만 명시적으로 구분.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 짧은 검사 범위.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
