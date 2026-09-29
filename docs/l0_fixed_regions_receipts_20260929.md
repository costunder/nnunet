# v2.2 고정 영역 캐시 독립 검토 수정 — 2026-09-29

후속 R3 검토 정정: [검증 표시 누락·메모리 단위·연속 배치 비용](l0_regions_r3_boundary_20260929.md). 아래 과거 측정의 `cache_tensor_bytes=25,925,496`은 실제 RAM이 아니라 저장된 `.pt` 파일 합계이다. 원시 보고서는 보존하며 새 도구에서 단위를 수정했다. 0.843초는 동일 GPU batch/이미 구성된 CSR의 update로 한정한다.

첨부된 독립 검토의 F01/F02/F03과 admission flag·RAM 재사용·전달본 누락을 수정했다. 기존 EZ-SP adapter, 초기 9개 검사, 과거 측정과 캐시는 보존했다. 이번 경로도 diagnostic-only이며 production runner를 교체하지 않았다.

## 수정과 검증 경계

- **F01:** 준비 fingerprint에 `l0_ezsp/encoder.py`, `tools/v222_review_contracts.py`, 실제 새 materialization, config/validation/hash 의존성을 포함했다. 격리 fixture의 파일 bytes 변경으로 fingerprint 변화를 검사했다. 매 update 소스 hash는 수행하지 않는다.
- **F02:** 실제 데이터 로더가 dataset index, record ID/file hash, donor/center, epoch/view와 CT·fine graph 전체 내용 hash를 남긴다. `prepare`가 별도로 받은 binding과 이를 대조한다. 중복 record, 바뀐 index/view, 변경된 CT를 거부한다. 캐시 형식은 `fixed_region_cache_receipt_v2`; 예전 캐시를 새 형식으로 재명명하지 않는다.
- **F03:** 준비 때 fine→1차와 1차→2차의 13종 quotient를 독립적으로 재계산해 확인한다. fine graph hash와 검증 사실은 준비 receipt에 남긴다. 저장·로드 시 2차 quotient를 재계산한다. 학습 batch에는 fine edge를 넣지 않는다. Receipt는 변경 감지용 hash이며 악의적인 작성자에 대한 서명이 아니다.
- **Admission:** N/E, bbox, role/shell cap을 저장된 partition에서 다시 계산하고, 분산 초과는 준비 audit와 함께 무결성 검사한다. pair별 실패와 준비 batch 전체의 보수적인 실패 flag를 분리했다. flag만 false로 바꾸는 것을 거부한다. reg/min_size/상한 변경 없음.
- **RAM:** 명시적 tensor byte 한도와 workers를 받는 `RegionResidentCache`를 추가했다. 첫 로드는 병렬 검증하고, 같은 요청은 검증된 batch를 반환한다. Tensor identity/version, shape/type/device와 metadata 변경을 검사한다. 변경 시 조용히 재사용하지 않고 오류를 낸다. 캐시 초과 시 sample을 버리지 않고 오류를 낸다. `.data`나 raw storage로 PyTorch version tracking을 우회하는 외부 변경은 지원하지 않는다. 한도는 보관 tensor 기준이며 일시적인 로드 메모리와 Python 객체 전체 RSS는 별도 실행 예산으로 확인한다.
- **측정:** L0 forward/L1·L2 objective 시간을 저장했다. 양쪽 source CT의 storage 수·내용 hash unique 수를 기록했다. 2차 입력 centroid 분산과 원래 fine 정규화 특징의 누적 분산을 구분해 기록한다. 누적 분산을 새로운 merge/admission 기준으로 적용하지 않았다.
- **전달본:** `hiercp_v221`, root `run_v222_v1_l0.py`, vendor PIN/LICENSE, 환경 버전을 포함한다. 별도 프로세스의 복사본 import/configuration/provenance/official vendor 검사로 누락을 확인한다.

## 실행 결과

GPU/CPU 회귀 **35개 통과, skip 0**. 이 숫자는 마지막 실행 한 번의 검사 수이며 앞선 32개나 과거 29개를 더하지 않았다. 합성 fixture 검사와 실제 CT 비용 진단은 별개다.

최종 원문: `validation/l0_fixed_regions_receipts_20260929/tests_final.txt`, `report.json`.
측정 소스: 같은 폴더의 `measured_source.zip`; report에 소스별 hash와 ZIP hash가 있으며 측정 시작/종료 bytes 일치 확인. ZIP은 측정한 Python/config 소스이며 vendor runtime PIN/LICENSE는 전달본 source에도 포함한다.

실제 기존 DEBUG CT 8pair 전부, physical/effective batch8, accumulation1, workers8, FP32, RTX5070Ti, GPU allocator 예산6GiB/RSS12GiB/240초. 로컬 기존 CNN checkpoint **step4/epoch1 DEBUG**를 사용했고 나머지 SAGE/L1/L2는 공통 seed42 초기값이다. 각 trial은 복제한 모델과 새 AdamW의 첫 update. Warmup1+교대3회. 장기학습·전체평가·production checkpoint 생성 없음.

| 항목 | Fine SAGE | 고정 영역 SAGE |
|---|---:|---:|
| Update 평균 | 1.440초 | 0.843초 |
| L0 forward | 0.513초 | 0.236초 |
| L1/L2 objective | 0.092초 | 0.102초 |
| Backward | 0.792초 | 0.440초 |
| 별도 8record refresh | 0.490초 | 0.247초 |
| Peak allocated | 0.791GiB | 0.342GiB |
| CNN 고유 source CT 수 | 7 | 7 |

Update 약41.4% 감소는 이 실행 안의 관측이다. **동시에 다른 프로젝트의 GPU 학습이 실행 중**이었다(시작 시 GPU 사용47%, 메모리8822MiB). 해당 작업은 변경하지 않았다. 깨끗한 단독 GPU benchmark, 서버/MIG 속도 또는 이전 0.614초와의 회귀 판단에 사용하지 않는다. Update는 loader/H2D·refresh/plan·production 저장 완료 시간을 제외한다.

별도 초기 비용: 준비7.234초, 저장3.029초, 첫 읽기/검증/collate6.336초. 이후 같은 CPU batch의 RAM 조회 **0.00431초**(3회 평균). 저장된 `.pt` 파일 합계25,925,496bytes(과거 tensor 캐시 표기를 정정). 첫 로드가 무료가 된 것은 아니며 전체 cohort RAM 수용량/epoch 시간은 측정하지 않았다. 보관한 첫 순차-load 진단도 삭제하지 않았다.

그래프는 동일: fine **52,666N / 2,761,602E** → scale1 **13,868N / 379,049E** → scale2 **13,849N / 378,284E**. 2차 추가 압축은 여전히 미미하다. 초기 bbox·분산·N/E 상한 위반을 그대로 보고했다. 분할의 의미적 품질, 서버 전체 support/validation 비용, CP 효용은 미검증이다. 이 결과를 3시간/epoch 해결 또는 production 가능 상태로 표시하지 않는다.

Basic CP, L1/L2, loss, 후보128, 원본 mask, 레이어/hidden/physical batch와 기존 observation 계약은 변경하지 않았다. Git push는 이 수정 작업에서 수행하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인된 고정 영역 구조 유지.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 같은 DEBUG8pair/workers8 비교.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사는 명시적으로 분리했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 짧은 검사 범위.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
