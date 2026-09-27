# v2.2 온라인 실행 경계 수정 — F01/F02/O01/D02

사용자가 전달한 `295598a1-62c4-4f1b-bfc3-b744e24eab07` 검토문 전체를 읽고 활성 호출 경로와 대조했다. 검토문이 지적한 설치본 우선 import와 scorer runtime 누락은 코드에서 확인되어 수정했다. 외부 sandbox 링크의 별도 ZIP/MD는 로컬 첨부가 아니므로 읽었다고 주장하지 않는다.

## 구현한 수정

| 항목 | 원인 | 이번 변경 |
|---|---|---|
| F01 | 상속받은 bank가 설치된 nnunetv2 raw engine을 먼저 import | 새 RankedBank에서 검토된 로컬 RawBankStore/apply_candidate를 직접 사용. 실제 파일 경로와 SHA256을 실행 출력·재개 계약에 기록 |
| F02 | 새 owner에 결정론/TF32/benchmark 설정 누락 | 모델 load와 매 recommend 모두 공유 GPU lock 안에서 기존 frozen_scoring_runtime을 적용. Python/NumPy/Torch CPU·CUDA RNG와 backend를 정상·예외 모두 복구 |
| F02 CUDA 시작 경계 | CUDA 초기화 뒤 환경변수 변경은 충분하지 않음 | launcher에서 CUDA 사용 전 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 설정. 이미 잘못 초기화했거나 다른 설정이면 오류 |
| O01 | segmentation launcher가 fresh만 지원 | `--resume`, 선택적 `--checkpoint`; 실제 파일 검사 후 선택한 파일만 load. 없으면 오류. plans/split/GNN/bank/선택 계약/로컬 engine/seed42/workers/250epochs와 결속 |
| O01 중단 복구 | upstream은 checkpoint 간격 동안 진행분 소실 가능 | 완료된 에포크마다 latest 저장. optimizer/scaler/logger/RNG/다음 epoch 복구. 중단된 epoch는 다시 실행하며 mid-epoch exact resume라고 부르지 않음 |
| D02 | 그래프 저장 가드가 native/선택 payload/checkpoint에 미적용 | 새 native preparation·runtime arrays·선택 payload·segmentation checkpoint 쓰기 전에 payload 추정+80GiB reserve 검사. 공간 부족은 오류이며 no-placement로 처리하지 않음 |

새 `tools/v22_online_native.py`는 로컬 prepare_case/prepare_candidate를 그대로 호출한다. native baseline 검사는 모든 CT/segmentation voxel을 bounded slab으로 비교한다. preparation arrays는 recipient당 한 번 별도 저장하고, 후보 graph 128개는 디스크에 저장하지 않는다. 선택된 donor/recipient payload와 receipt만 요청된 이벤트에 대해 저장한다. 같은 이벤트는 receipt를 재사용한다.

온라인 저장이 무한히 작다는 뜻은 아니다. 많은 서로 다른 쌍이 요청되면 selected payload가 증가한다. 원본 CT, 전체 training graph cache, native runtime/preparation, selected payload, segmentation checkpoint를 별도로 계산해야 한다. `tools/v22_online_storage.py --bank <index.json>`은 관측된 case/pair의 최대 추정량과 전체 경우의 수를 사용한 경험적 전망을 출력한다. 관측치가 없으면 null이며 0으로 취급하지 않는다. 최신 geometry의 엄밀한 상한이 아니며 다른 프로세스의 동시 디스크 쓰기까지 예약하는 파일시스템 quota도 아니다. 기존 파일 자동 삭제·샘플 생략·작은 모델 fallback은 없다.

## 재개 명령과 호환성

아래 환경 변수는 실제 production bank/results 경로와 측정된 worker 수를 가리켜야 한다. 새 online bridge의 체크포인트가 있는 같은 결과 폴더에서 실행한다.

```bash
python -u tools/train_v22_online_rank.py --bank "$ONLINE_BANK/index.json" \
  --results "$SEGMENTATION_RESULTS" --workers "$MEASURED_NATIVE_WORKERS" \
  --resume --check-only
python -u tools/train_v22_online_rank.py --bank "$ONLINE_BANK/index.json" \
  --results "$SEGMENTATION_RESULTS" --workers "$MEASURED_NATIVE_WORKERS" --resume
```

특정 저장 파일은 `--checkpoint "$SEGMENTATION_CHECKPOINT"`로 지정한다. 지정하지 않으면 final → latest → best 순으로 실제 존재하는 파일을 고른다. 선택 파일 SHA256을 출력하고 실행 직전 다시 확인한다. 파일이 없거나 계약이 다르면 fresh로 전환하지 않는다. fresh는 `--resume` 없이 새 폴더에서만 가능하다. `--check-only`는 경로·계약 검사이며 capacity 또는 학습 성공 증거가 아니다.

이 재개 계약이 없는 과거 segmentation checkpoint를 새 계약이라고 재표시하지 않는다. 기존 GNN checkpoint/재개 경로와 별개다. 새 bridge 파일 hash가 바뀌었으므로 이전 online catalog도 새 코드로 자동 인정하지 않는다. 올바른 production final GNN과 native 준비에서 새 catalog를 생성해야 한다. catalog는 작은 목록이며 전체 그래프 캐시를 재생성하는 명령이 아니다.

## 검증 범위

기계 판독 결과는 `validation/v222_r6/online_review_fixes_20260927_DEBUG.json`, 실제 CT 실행 결과는 `work/v22_online_fixes_20260927_DEBUG/actual_CT_pass/result.json`에 기록한다. 단위/회귀와 실제 CT 결과를 합쳐 전체 cohort 합격으로 해석하지 않는다.

**실제 실행 결과:** 65개 고유 단위/회귀 테스트 통과. 마지막 checkpoint 필수 상태 보강 후 관련 7개를 다시 통과했으며 중복 합산하지 않는다. 실제 CT 통합 재실행도 통과했다. 128개 중 index57(원래 rank1, 점수0.54647839, 간 포함률0.95059122, 종양 겹침0)이 선택됐다. 최종 native CT의 독립 oracle 최대 절대 오차는 **0.0**, segmentation은 exact이며 원본 native baseline도 CT 오차0.0/seg exact다. 이번 케이스는 원래1위가 유효했다. 원래1위 탈락 후2위 선택과 None 원본 유지는 별도 synthetic native 테스트로 확인했다.

전체 GNN은 **5,550,806 parameters**, 128후보를 3회 채점(반복·GT 대조 포함)했다. 새 그래프 생성·native 준비·독립 전처리 oracle을 포함해 **265.61초**, 프로세스 CUDA peak allocated **271,563,776 bytes**, sampled peak RSS **5,168,508,928 bytes**였다. GPU 전체 점유량이나 full-support/segmentation 공존 메모리가 아니다. graph 생성은 worker1/2/4의 서로 다른 후보 wave를 측정 후 해당 실행에서 worker2를 선택했다. 동일 입력 microbenchmark나 최적성 증명은 아니다. 저장은 실행별 약**0.7314GiB**이며 첫 실패 검사까지 합쳐 약1.46GiB다. 전체 캐시나 전체 모델 학습을 추가 생성하지 않았다.

한 관측 case와 한 selected pair의 **쓰기 추정량**을 전체105 recipient/55,335 pair slots에 확대하면 native 약153.64GiB + selected pair 약81.43GiB = 약235.07GiB다. 실제 전체 저장량 측정이 아니며 두 배 쓰기 여유·shared source 중복도 포함한 경험 추정이다. 이 수치 때문에 온라인 저장을 별도 계산하고, 매 쓰기 전 admission을 유지한다.

- 설치 모듈 없음/같은 구현/다른 구현 모두 로컬 engine 선택을 검사한다.
- 정상 및 예외 시 CPU/CUDA RNG/backend 복구, 늦은 CUDA 환경 설정 거부를 검사한다.
- 실제 tensor와 Adam state 저장→복구, epoch cursor, RNG, 계약 변경/파일 누락 거부를 검사한다. CLI가 선택된 실제 checkpoint를 load하는 경로도 검사한다. 전체 nnU-Net 장기 재개 실험은 아니다.
- 디스크 부족 반례에서 실제 writer 호출 전에 거부되는지 검사한다.
- 실제 CT 진단은 liver_31 + donor liver_73 component1, **production 규칙의 128개 후보와 전체 GNN 구조**를 사용한다. 기존 DEBUG final checkpoint의 **support8, physical batch2**이며 production 설정을 바꾸지 않는다. 오래된 실제 native 준비를 명시적 DEBUG fixture로만 사용하며 production admission은 우회하지 않도록 그대로 유지한다.
- 첫 실제 실행은 owner/RPC/선택/저장 후 DEBUG loader 생성자의 초기화 누락으로 실패했다. 실제 base 생성자를 사용하도록 검사 코드를 수정했다. 실패 로그와 데이터는 보존한다.
- 최종 입력 비교는 표준 영상 증강 직전 native CT/segmentation을 대상으로 한다. 전체 segmentation forward/backward 및 전체 support 공존 검사는 아니다.

GNN core83/runtime20은 `ea702fd` 기준선과 byte hash로 대조한다. 모델 깊이·폭, L0/L1/L2, 목적함수, graph/candidate 수, 전체 데이터, production physical batch는 수정하지 않았다. 기본 loss의 pair 가중 편차, detached L0 reference, CE/rank 관계, tumor appearance shortcut, fixed-donor 학습 위험은 별도 연구 과제이며 이번 연결 수정으로 해결됐다고 주장하지 않는다.

검토의 D01(full support와 segmentation 동시 VRAM)은 **미검증 capacity 문제**다. 이번에 OOM을 재현했다고 표현하지 않는다. 로컬 GPU에서 다른 프로젝트가 동시에 실행 중이므로 이번 경과 시간은 독점 처리량이 아니다. 전체 support/current production artifact·worst-profile·segmentation optimizer 공존 G3/G4, no-CP/valid-random-CP/proposed-CP 효용 G5는 남아 있다. 전체 cache·GNN40epoch·segmentation250epoch·전체 평가는 실행하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] 기존 batching과 측정 기반 병렬 graph 생성 경로를 보존했다. 전체 공존 physical batch 검증은 미완료다.
- [x] GPU, CPU, RAM 상태와 실제 DEBUG 자원을 확인했다.
- [x] OOM 회피를 위해 모델을 축소하지 않았다.
- [x] DEBUG 초기화/데이터와 production 설정을 분리했다.
- [x] production에 dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 기존 학습 forward/loss/gradient/optimizer 경로를 변경하지 않았다. 이번 통합 검사는 추천→최종 native 입력 범위다.
- [x] 실행 설정과 변경·미검증 사항을 보고했다.
- [x] DEBUG와 전체 학습·평가를 구분했다.
