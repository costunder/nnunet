# 비교 실험 실행·진행률 점검

대상: `selected`, `native`, `native_fixed`, `native_listwise`의 기존 10 mm 실험.
학습 문제당 8후보, 검증 문제당 129후보, 원본 두 view, 후보 순서·GT·loss·모델·40 epoch·측정된 physical batch와 worker 수는 유지한다.

## 확인한 문제와 변경

| 문제 | 변경 |
|---|---|
| epoch 학습 표시가 끝나면 검증 중임을 알 수 없음 | 첫 데이터 읽기 전부터 검증 개수·비율·단계·단계 경과시간·ETA 표시 |
| 긴 진행률과 주기적인 캐시 로그가 줄을 깨뜨림 | 터미널 폭 안의 짧은 표시, 비터미널은 제한된 주기의 줄 출력. 캐시 상세는 JSONL, 오류는 계속 표시 |
| 전체 검증 종료까지 점수 표시가 없음 | 관측된 문제의 patient-macro 중간 MRR/top1/pair-win 표시. 중간 값과 전체 검증 결과를 구분 |
| 검증의 CPU 입력 준비와 CUDA 추론이 순차 실행 | 다음 CPU 배치 한 개를 준비하면서 현재 배치를 추론. 원래 순서와 cursor 유지 |
| 후보마다 전체 CT에 `label == 1` 수행 | 같은 원본 patch를 먼저 추출한 뒤 `== 1`. 경계 패딩도 동일 |
| 후보별 두 sampled graph view를 Python 루프로 순차 생성 | 기존 worker 수로 후보별 두 view를 병렬 생성. 원본 builder·시드·노드·엣지·출력 순서 유지 |
| 화면의 step 시간에 데이터 대기가 빠짐 | 입력 구성, 실제 대기, forward/backward/optimizer, 체크포인트 시간을 구분해 기록 |

원본 CNN은 이미 source 특징과 두 view의 CNN map을 공유하고, GNN은 PyG Batch를 사용한다. 두 번째 graph view는 원본 consistency와 추론 평균에 쓰이므로 제거하지 않았다. Physical batch와 validation GPU chunk는 기존 측정값을 유지한다. 이 패치는 VRAM을 강제로 채우거나 새 GPU를 할당하지 않는다.

## 기존 체크포인트 재개

기존 `u_bridge_training.py`, `comparison_training.py`, 입력 계약 파일은 수정하지 않는다. 원본 소스 해시와 실험 manifest를 검증한 뒤 별도의 실행 모듈을 바인딩한다. V1.9의 loss·체크포인트 정책은 기존대로 적용한다. 모델·Adam·scheduler·AMP scaler·RNG·epoch/batch cursor를 복원하며 새 실험으로 초기화하지 않는다.

`execution_overrides`에 실제 적용한 새 실행 파일들의 해시를 기록한다. 네 실험의 output/lock/checkpoint는 계속 각각 소유한다. 실행 중인 프로세스에는 변경이 자동 반영되지 않는다. 다른 실험을 종료하거나 기존 캐시를 삭제할 필요는 없다.

## 상태와 병목 확인

```bash
python tools/report_comparison_runtime.py --experiment /home/aicompetition06/Medical/experiments/v19_native_listwise_m10_seed42 --arm native_listwise
```

이 명령은 checkpoint를 로드하지 않고 다음 기록을 읽는다.

- 실험 arm 폴더의 `progress.json`: 최근 단계·개수·중간 점수·갱신 시각. 오래된 기록은 프로세스 생존 증거가 아니다.
- `update_timing.jsonl`: loader 대기를 제외한 step과 포함한 전체 batch 시간, 실제 CUDA 구간, 저장 비용.
- `validation_timing.jsonl`: CPU 구성 시간과 실제 대기, CUDA 추론, 저장 시간. 미리 준비한 시간은 GPU 작업과 겹칠 수 있으므로 더해서 전체 시간으로 취급하지 않는다.
- 실험 root의 `input_timing.jsonl`: case/fields, region, source, 후보 metadata, canonical graph, sampled views, 나머지 hierarchy/collation 비용.
- `curve.jsonl`·`validation_epoch_*.json`: 완료된 전체 검증의 점수와 best 판단.
- `preparation_reuse.jsonl`: 전체 호출 이력의 cache 재사용·miss·field 상태. 누적 시간을 한 epoch 비용이나 ETA로 취급하지 않는다.

현재 서버의 NFS 처리량, 네 실험 동시 실행 시 경합, 전체 cohort의 추가 프리패치 RAM peak는 로컬 검사로 확정할 수 없다. 서버 실측으로 추가로 확인해야 한다. 같은 seed라도 CUDA 집계 연산의 실행별 수치 차이는 입력 텐서의 동일성과 별도로 검사한다.

## 검증 범위

별도 실제 CT/CUDA DEBUG 검증을 사용했다. 원본 10,434,532 parameter, physical source batch 2, worker 4의 보존된 DEBUG fixture로 CUDA optimizer update 2회와 전체 129후보 검증 3회를 실행했다. 1,085개 학습 parameter tensor의 finite gradient, 완료 체크포인트 재개 시 추가 update 0회, 실제 실행 wrapper, 원본 파일·캐시 보존을 확인했다. 이 수치는 production batch/worker 설정을 바꾸라는 뜻이 아니다.

원본 순차 입력과 변경 입력의 8후보 학습 배치 및 129후보 검증 배치는 모든 tensor 값·shape·stride·PyG 연결 메타데이터가 일치했다. 완료한 2-update 가중치를 과거 별도 CUDA 실행과 비트 단위로 비교하는 추가 검사는 불일치했다. 이를 통과했다고 처리하지 않았다. 동일 입력·가중치·RNG의 반복 추론을 추가로 검사한 결과, 원본 경로 자체에서도 최초 L0 GNN block부터 작은 수치 차이가 있었다. CNN 출력은 동일했다. 최종 학습 가중치·점수의 비트 단위 동일성은 보장하지 않는다.

수정 후 로컬 DEBUG의 검증 CPU 준비 3회 합계는 53.54초였고, 그중 sampled graph views가 40.49초(75.62%)였다. 마지막 검증은 입력 대기 17.27초, CUDA 추론 12.33초였다. 해당 fixture는 검증 source problem이 하나라 다음 검증 배치와의 overlap 처리량을 측정한 결과가 아니다. 단위 검사에서 다음 배치 준비가 현재 배치 소비와 겹치면서 순서·실패·RNG를 보존하는지 별도로 확인했다. 남은 CPU graph 생성 비용과 서버 NFS 경합이 모두 사라졌다고 주장하지 않는다.

전체 production 재학습·전체 cohort 속도 측정·CP 추천 품질 개선은 수행하지 않았다. 세부 검사 결과는 `validation/comparison_runtime/verification.json`에 기록한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 모델 축소보다 입력 준비·동기화·캐시·병렬화 병목을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 원본 forward·loss·gradient·optimizer 경로를 유지했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
