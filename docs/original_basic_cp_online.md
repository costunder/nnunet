# 원본 Basic CP — 온라인 실행만 변경

사용자 정정: 원본 Basic CP에 크기 제한, 고정 후보 풀, 외부 환자 donor를 추가하지 않는다. 원본의 정책을 유지하고 CT/GT를 파일로 미리 생성하는 시점만 nnU-Net 학습 데이터 로딩 시점으로 옮긴다. 구현은 `basic_cp_online/`, 진입점은 `run_basic_cp_online.py`다. 기존 v1 파일과 과거 실험 결과는 보존한다.

## 원본에서 유지하는 규칙

- 같은 환자의 label=2 연결요소 전체에서 균등 무작위로 하나 선택한다. 20 mm 이하 제한, 큰 종양 축소, 다른 환자 donor, GNN, 128개 후보 풀은 없다.
- pad=2의 원래 CT·마스크 patch를 사용한다. 해당 환자의 간 좌표를 섞고 최대 4,000개를 검사하며 처음 유효한 위치에서 멈춘다. 후보 4,000개는 새 축소 제한이 아니라 원본의 탐색 상한이다.
- 간 coverage 0.85, 기존 종양과 6-neighbor dilation 2 voxel 간격, 원본의 nominal center distance 12 voxel 설정을 유지한다.
- hard paste, HU scale 0.95–1.05, shift −5–5, 새 종양 1개 시도를 유지한다. 기존 원본에 없는 50% 적용 gate를 추가하지 않는다. 각 학습 case 방문마다 새 선택을 수행한다. 방문별 seed를 기록한다.
- 종양 또는 간이 없거나 4,000개 탐색에서 유효 위치가 없으면 증강하지 않은 데이터를 사용하고 사유를 기록한다. 다른 donor를 재추출하지 않는다.

## 원본의 알려진 오류를 숨기지 않음

원본 batch는 tumor mask를 uint8로 만든 뒤 `~mask`를 사용한다. 그 결과 0/1의 논리 반전 대신 255/254가 되어 거리 계산이 실제 종양으로부터의 거리가 아니다. 이 온라인 원본 재현 경로는 그 동작까지 보존하며 `literal_reference_uint8_inversion_known_defect`로 기록한다. 비교 편의를 위해 boolean 거리 계산으로 조용히 교체하지 않는다. 올바른 거리 계산으로 수정한 실험은 별도 실험으로 구분해야 한다.

원본 함수 중 bounding box와 무작위 좌표 탐색 함수를 SHA 검증 후 그대로 사용한다. 출처 SHA는 Git/Windows CRLF 차이를 제외한 LF 정규화 SHA `986fd6afb95b94c70614ac02d2b9ced776cbd680c5e70d4dd11e20148383bd70`다. 원본 파일 자체는 수정하지 않는다.

## 온라인 연결과 누수 경계

`prepare`는 전체 outer-train의 불변 CT/GT·연결요소·금지 영역·native 변환만 저장한다. 종양 크기를 선별하거나 donor/붙일 위치를 미리 정하지 않는다. 학습 도중 `Dataset.load_case`에서 원본 규칙으로 새 donor와 첫 유효 위치를 선택한다. CT는 요청된 native crop만 정확한 raw-paste→nnU-Net 전처리 결과로 계산하여 full CT를 매번 복사하지 않는다.

증강된 전체 GT에서 nnU-Net의 기존 foreground 위치 추출을 다시 수행한다. 기존 기본 loader의 crop, 표준 augmentation, physical batch, 모델을 유지한다. 붙인 종양 주변으로 crop을 강제 이동하지 않는다. 기존 native 변환으로 원본 placement를 정확히 표현할 수 없는 경우에는 명시적 오류를 내며 후보·donor·크기를 대체하지 않는다.

학습 dataset만 wrapper에 연결한다. validation dataset과 loader는 기존 nnU-Net 그대로다. 분할·training-only planning·원본 파일 동일성·준비 artifact SHA를 확인하며 held-out CP 요청을 raw volume 읽기 전에 차단한다. 학습 checkpoint나 GNN bank는 필요하지 않다. v2.1 공통 donor GNN은 별도 실험으로 남으며 이 Basic CP와 donor 조건이 같다고 주장하지 않는다.

## 실행과 검증 범위

전체 불변 입력 준비:

```powershell
.\.venv\Scripts\python.exe run_basic_cp_online.py prepare --native work/local_v21_5070ti_20260919/native/native.json --output work/original_basic_cp_online_full
```

완료된 manifest로 `train --manifest <index.json> --output <새 결과 폴더> --workers <실측 worker 수>`를 실행한다. native ResEncM 계획과 batch를 그대로 사용하며 250 epoch다. 기존 환경에 trainer를 덮어쓰지 않고 결과 폴더에 별도 nnU-Net runtime을 만든다. worker 수는 임의의 작은 기본값을 넣지 않고 필수 인자로 받는다. 실제 전체 preparation·병렬 loader 처리량·co-resident GPU batch 측정과 250 epoch 본학습은 아직 실행하지 않았다.

DEBUG 8개에서 원본 함수 실행 결과와 온라인 CT/GT 완전 일치(12개 seed, 20 mm 초과 종양 포함), 무종양·무간·배치 실패 시 원본 유지, 매 방문 새 위치, literal 거리 계산 동등성, native 전처리된 전체 CT/GT 및 padded crop 동등성, 실제 표준 nnU-Net batch 생성, validation wrapper 배제를 확인했다. 합성 입력이며 전체 의료 학습/성능 결과가 아니다. 추가 실데이터 원본 비교 증거는 `work/original_basic_cp_online_20260919/`에 별도로 기록한다.

실제 학습 CT `liver_2`의 전체 볼륨 대조도 통과했다. seed 42에서 자기 환자의 component 1(등가 직경 25.3166 mm, 14,131 raw voxel)을 크기 축소 없이 선택했고 첫 제안 위치가 유효했다. 원본 스크립트를 실제 실행한 출력과 새 정책의 전체 raw CT·GT가 정확히 일치했다. 원본 파일 SHA도 변하지 않았다. 소형 source가 없다는 이유로 제외하지 않는다. 소요 89.38초, 비교 검사 peak RSS 13,686,300,672 bytes이며 원본과 새 출력 전체를 함께 보관한 DEBUG 검사 자원이다. 온라인 loader 처리량으로 해석하지 않는다. 증거: `work/original_basic_cp_online_20260919/debug1/verification.json`.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. Basic CP에는 그래프가 없다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 4,000회는 원본 탐색 계약이다.
- [x] physical batch size와 병렬화 가능성을 검토했다. 기존 native batch 및 다중 worker loader를 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 확인했다. 새 Basic 전체 처리량 측정은 미완료다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한다. native CT는 crop만 계산한다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 실제 결과로 사용하지 않았다.
- [ ] 실제 의료 Basic CP → nnU-Net forward/loss/backward/optimizer 전체 연결 실행은 아직 검증하지 않았다. 데이터 loader까지 검증했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
