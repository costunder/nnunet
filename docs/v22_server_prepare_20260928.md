# 서버 전체 캐시 준비 전용 실행

사용자 서버 출력: ece-agpu16, Python3.10.18, torch2.6.0+cu118, PyG2.7.0, 할당 GPU6 MIG1g.10gb(9.5GiB). 설치 부모의 직접 종료 호출 목록은 빈 목록이지만 간접 호출/OS 신호 안전성까지 검증한 것은 아니다. 서버 전체 CPU128/RAM 약2TiB 수치를 개인 전용 할당으로 간주하지 않는다. 기존 준비 scheduler는 affinity/cgroup과 실제 처리량/RSS를 확인한다.

발견된 기존 cache는 `HierCP-v222-r6/work/v222_mig10gb_r6_scanfix/paired_cache/index.json`이다. 설정과 base는 같지만 data.py 두 개, placement.py, record_binding.py, v1_cache.py, v1_local.py가 달라졌다. 0681399→e1e34bf diff에는 donor anchor를 보존하는 padding, PlacementSpec footprint 사용, CT/placement/graph 내용 결속이 포함된다. 기존 graph에 새 hash만 붙일 수 없다. 원본 CT와 기존 결과는 보존하고 새 cache를 만든다.

사용자 실측은 graph14,102개16.59GiB, shared source527개0.54GiB, donor527개2.22GiB다. 기존 category별 최대 파일 크기×개수 추정은44.68GiB, 여유362.93GiB, 기존 reserve80GiB다. 추정+reserve124.68GiB를 넘는 여유가 있다. 새 관측 목록이 생성되면 실제 개수로 다시 추정하고 writer 직전에 admission한다. 새 기하에 대한 보장 상한은 아니며 writer도 기존 여유 공간 검사를 유지한다. 별도 사용자 quota를 조회했다고 주장하지 않는다.

## 실행과 검증 범위

`tools/run_v22_server_prepare.py`는 foreground에서 기존 구현을 순서대로 호출한다.

1. `v1_server.observations`: 전체131case 원본 CT/GT 검사와 관측 metadata 작성.
2. `v22_cache_storage.make_plan/admit`: 기존 cache는 파일 크기 참조로만 사용. 최신 관측 목록·새 출력 경로에 결속된 용량 계획 및 reserve 검사.
3. `v222_prepare_optimized.prepare`: 기존 full-cohort donor/graph 생성·병렬 측정·부분 파일 보존 구현 그대로 사용.
4. `preparation_complete.json` 기록 후 종료. GNN/nnU-Net train 함수나 background worker를 호출하지 않는다.

표준 출력은 `console.log`에 전부 보존하고 JSON 진행 이벤트만 화면 tqdm에 반영한다. 학습 viewer를 붙이거나 tail할 필요가 없다. Ctrl+C는 foreground 준비를 중단하며 실행 중인 CPU 작업이 정리될 때까지 기다릴 수 있다. 기존 산출물은 보존하지만 이 CLI가 자동 재개를 제공한다고 주장하지 않는다. 기존 output 경로는 덮어쓰지 않고 거부한다. 실패 시 명시적 오류/실패 기록을 남기며 작은 모델/데이터로 fallback하지 않는다.

로컬에서 orchestration 단위검사4개와 CLI help/구문 검사를 수행했다. 용량 검사 실패 시 graph builder 미호출, 저장계획 전달 및 학습 미시작 표시, 기존 파일 덮어쓰기 거부, lifecycle/progress 분리 표시를 확인했다. 작은 fixture로 검증했으며 실제 CT 준비나 GPU 검증으로 보고하지 않는다. 이번에는 새 전체 cache를 로컬에서 만들지 않았다. model/core83/runtime20/online15 파일은 변경하지 않았다.

이 명령의 완료는 **캐시 준비 완료**다. 전체 support G3, production 초기화/worker/compile/새 프로세스 재개 G4, 효용 G5 통과가 아니다. 장기 학습은 아직 시작하지 않는다. 이 실행기 추가 때문에 이미 최신 계약으로 생성된 cache를 다시 만들 필요는 없다.

## 서버 병렬 출력 실패와 수정

첫 서버 실행(e3259ca)은 raw_inventory 화면29/131에서 `JSONDecodeError: Extra data`로 실패했다. 모델/CT/graph 오류가 아니라 새 ProgressLog의 공유 partial buffer 오류다. `print()`는 본문과 줄바꿈을 서로 다른 write로 보내므로 write별 lock만으로 한 print의 원자성을 보장하지 못했다. 최초4개 검사는 단일 스레드만 다뤘고 이 조건을 놓쳤다.

기존 e3259ca 원본 ProgressLog에 실제 두 스레드와 barrier를 적용해 두 JSON 본문이 줄바꿈 전에 합쳐지는 동일 `Extra data` 오류를 재현했다. 수정본은 스레드 객체별 미완성 버퍼를 유지하고 완성된 줄만 lock 안에서 로그와 parser에 함께 전달한다. Thread ID 재사용으로 다른 스레드의 조각을 물려받지 않도록 Thread 객체를 키로 쓴다. 종료 시 미완성 일반 텍스트도 보존한다. 실제 malformed event를 조용히 무시하지 않는다.

수정 후 총8개 단위검사 통과: 기존4개에 정확한 서버 교차 출력, 16개 실제 스레드×100개의 강제 JSON 조각 교차(총1600개 로그의 내용/누락/중복 검사), 미완성 텍스트 보존, malformed event 명시적 실패를 추가했다. 로컬 CPU 동시성 검사이며 서버131case/전체 그래프 완료나 G3/G4 통과는 아니다. 출력 계층만 변경하고 기존 core83/runtime20/online15는 보존한다.

실패한 `work/v22_full_prepare_20260928`은 그대로 보존한다. graph 생성 전 관측 단계에서 실패했으므로 새 `work/v22_full_prepare_20260928_logfix`에서 관측부터 다시 실행한다. 이미 처리된 case를 이어받는 exact resume라고 설명하지 않는다. 생성 중이던 기존 CPU 작업은 예외가 전파되기 전에 executor가 정리하며 별도 종료 신호는 보내지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size는 변경하지 않았다. 기존 준비 scheduler의 병렬 측정을 유지했다.
- [x] GPU/CPU/RAM 사용자 실측과 기존 scheduler의 자원 확인 범위를 구분했다.
- [x] OOM을 가정해 모델을 축소하지 않았다.
- [x] 단위검사 fixture와 전체 데이터 준비를 구분했다.
- [x] dummy/placeholder/random fallback을 production에 추가하지 않았다.
- [x] 기존 forward/loss/gradient/optimizer 구현을 변경하지 않았다. 새 실행기는 준비 전용이다.
- [x] 실제 설정과 변경 및 검증 범위를 보고했다.
- [x] 준비 완료·smoke·전체 학습/평가를 구분했다.
