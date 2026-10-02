# v2.2 희소 L0: 실제 간 내부 경로를 보존하는 relay DEBUG 구현

2026-10-02. 기존 대표점만 연결한 `V1RelationalSparseL0` 결과와 구현은 보존하고, `RelayedV1SparseL0`를 별도 DEBUG 경로로 추가했다. 이번 수정은 원본 대표점 사이의 문맥 연결이 끊기는 문제를 대상으로 한다. 전체 학습, CP 효용 검증, 서버 학습 가중치의 성능 개선 결과가 아니다.

## 확인한 문제와 수정 범위

기존 희소 경로는 CNN·공간 특징으로 고른 대표점만 남긴 뒤 6 mm 이내의 가까운 세 context 이웃을 연결했다. 실제 CT의 XY spacing은 약 0.676/0.713 mm, Z spacing은 5 mm여서 가까운 XY 이웃만 선택되고 유효한 Z 연결도 제외됐다. `liver_66`의 P 5개와 U 128개 전체에서 pair별 weak connected component가 19–53개, 고립 노드가 10–36개였다. 따라서 대표점 수·엣지 존재만으로 문맥 그래프가 완성됐다고 판단할 수 없었다.

v1의 `hiercp/sample.py`는 seed 이후 필수 이웃과 hop 이웃을 보존한다. 이번 경로도 **대표점 수를 최종 node cap으로 취급하지 않고 연결에 필요한 중간점을 보존**한다. v1의 큰 fine graph를 복원하거나 seed 개수·모델 크기·반경을 줄인 변경은 아니다.

변하지 않은 계약은 다음과 같다.

- 원래 CNN 8 conv, channels `[12,24,32]`, native 해상도와 organ-only 입력, 같은 전체 CNN feature pool.
- 48/96/192개의 context seed를 각 donor/recipient branch에 유지. 각 밀도는 near/mid/wide 16/32/64개씩이며 원래 query와 seed의 좌표·특징·역할을 그대로 보존한다.
- 방향과 역할을 구분하는 8종 mean-SAGE 관계, 3층·hidden 128D, 역할·shell readout과 기존 paired fusion→128D→L1/L2.
- 전체 네트워크의 parameter/trainable parameter 수는 **1,785,558**로 두 경로가 같다. 이 수치는 L0 단독 크기가 아니다.
- 원래 P/U 정의, same-donor 입력, ranking/CE/alignment loss와 normalization, 후보 U 128개, 원본 mask 및 Basic CP.
- physical batch 32, accumulation 1, data parallel worker 1, effective batch 32. 마지막 전체-case batch의 5개 잔여 관측도 버리지 않는다.

P는 실제 관측된 적격 종양 anchor이고 U는 기존 미관측 비교 중심이다. U를 CP 부적합 GT로 바꾸거나 P를 특정 donor의 compatibility GT로 재정의하지 않았다. 종양 주석은 정답과 시각화 overlay에 사용하며 새로운 relay의 선택 근거로 사용하지 않는다.

## 실제 sampling과 edge 구성

`l0_sparse_feature/relay_sampling.py`는 organ 내부이며 모든 CNN scale에서 지원되는 native voxel pool만 사용한다. native 6축 이웃의 각 물리적 한 step이 6 mm 이내인지 확인하고, 원래 3 mm query sphere의 유효 voxel들을 root로 batched BFS parent forest를 구성한다. 전체 fine voxel들은 이 sampling substrate이며 모두 SAGE 노드가 되지는 않는다.

선택된 seed에서 실제 parent 경로를 따라 필요한 relay를 유지한다. 중간 경로를 압축할 때에는 직선 거리 대신 **유효 native 경로의 누적 물리 길이 ≤6 mm**를 강제한다. 기존 nearest-three context edge에 이 경로의 양방향 context edge를 합치고 동일 방향·관계의 중복만 제거한다. 필수 경로는 nearest-three pruning 이후에 추가되므로 최종 context incoming degree가 세 개를 넘을 수 있다. 이를 세 개 제한을 지킨 결과로 보고하지 않는다.

query–context 8 mm, context 교차 관계 5 mm, donor query→recipient context 8 mm 등 이전 typed relation 반경은 유지한다. 원래 관계에 새 MST, kNN 재연결, out-of-radius bridge, 반경 확대, node drop, record skip, 작은 graph fallback을 넣지 않았다. native 6축 substrate에서 접근할 수 없는 seed는 보존하면서 원인과 개수를 진단으로 남긴다. native6 접근 실패가 전체 6 mm radius graph에서도 불가능함을 증명하는 것은 아니다.

`verify_retained_paths()`는 mandatory edge마다 실제 parent chain을 GPU에서 다시 확인한다. 역할·scene·padding·eligible mask·좌표 결속, native axis 한 step, parent depth 감소와 cycle, 누적 길이와 step count를 검사한다. 잘못된 witness 길이나 임의 parent/cycle/좌표를 받아들이지 않는다. voxel hole을 직선으로 가로지르는 선을 검증된 간 내부 경로로 표시하지 않는다.

동일 physical batch 안에서 반복되는 `(CNN crop ID, native anchor)`의 geometry와 support가 정확히 같은지 검증하고 FPS/BFS를 한 번만 계산한다. 결과는 원래 donor/recipient 64 scene occurrence와 32개 disjoint pair로 확장한다. 샘플 합치기, physical batch 축소, gradient accumulation 대체가 아니다.

## 실제 CT·초기 가중치·원본 보존

최종 실행은 `work/relay_sparse_CT_DEBUG_20261002_r2`에 별도로 저장했다. 기존 r1 및 기존 relational 결과를 덮어쓰지 않았다.

- recipient `liver_66`, P 5개＋U 128개＝전체 133개 관측, 해당 case 사용률 100%.
- donor `liver_1`, 원래 assignment의 component 1. assignment SHA-256 `b7347a7d383f7010a6c114874121f8a2bb60d3910dd807ed4d6e4de0f433a19f`.
- CNN/fusion/L1/L2는 로컬 DEBUG saved step 4 snapshot을 정확히 복사했다. checkpoint SHA-256 `9b36e85a57819042026b152e7bcc7236ed9ad0b372fa32b5d5c321d74b50c1f6`. 서버에서 학습된 weights를 사용한 것이 아니다. 그래프 모듈은 seed 42의 새 초기값이며 exact resume가 아니다.
- 두 경로×세 밀도의 초기 전체 model hash가 동일하다. 신규 relay 추가는 parameter를 추가하지 않는다.
- 전체 DEBUG loss context 139관측의 기존 schedule, normalization 및 donor/query group 제외를 유지했다. 원래 memory 8개 모두 새 L0로 재인코딩하고 group 제외 후 6개의 eligible support와 teacher를 각 경로에서 다시 만들었다. 옛 mean-memory를 새 그래프 query와 섞지 않았다.
- 실행 전후 11개 구현 파일 hash가 같고 원본 입력·memory·source·schedule 보존 assertions가 통과했다. production 121개 source와 기존 `REFERENCES.md`, `code.txt`, `gpt_handoff.md`의 dirty 변경은 보존한다.

원시 report와 native 좌표·anatomy payload는 로컬 `work`에 보존한다. Git용 `validation/relay_sparse_20261002/actual_CT_DEBUG_summary.json`은 원시 report hash와 수치·결속 기록을 담으며 CT pixel/anatomy, checkpoint를 포함하지 않는다.

## 전체 133관측의 그래프 결과

아래 N/E는 donor와 recipient를 합친 실제 pair 하나의 범위이다. seed 수는 branch 하나의 context seed 수이며 최종 node cap이 아니다. 이전 모델도 같은 8종 typed relational 모델이다.

| branch seed | 이전 N / E | relay 추가 후 N / E | 추가 relay/pair | 이전 weak components / isolates | 새 weak components / isolates |
|---|---|---|---|---|---|
| 48 | 98 / 245–289 | 256–300 / 1,183–1,365 | 158–202 | 19–32 / 13–27 | 1 / 0 |
| 96 | 194 / 589–682 | 439–524 / 2,254–2,581 | 245–330 | 21–44 / 12–36 | 1 / 0 |
| 192 | 386 / 1,338–1,492 | 741–902 / 4,028–4,746 | 355–516 | 19–53 / 10–35 | 1 / 0 |

세 밀도×133개＝399개 새 pair graph가 모두 weak component 1개, 고립 노드 0개였다. 원래 seed를 하나도 버리지 않았고 native6 unreachable seed는 0개였다. 전체 동일 feature pool을 대상으로 계산한 공간/CNN/joint coverage가 relay 추가 뒤 나빠지지 않았음을 확인했다. coverage deficit은 종양에 유효한 정보 손실률이나 CP 추천 정확도가 아니다.

**Weak component 1개는 3층 SAGE에서 모든 노드의 정보가 query로 도달한다는 뜻이 아니다.** 독립 감사한 각 밀도별 표시 8개에서 recipient query의 3-layer ancestor 수 평균은 이전 22.875/25.5/23.75개에서 새 48.5/39.75/37.5개로 늘었다. 그러나 wide-role→recipient-query 3-hop ancestor는 새 48 경로의 표시 8개를 합쳐 1개, 새 96/192에서는 0개였다. 모든 wide node가 3층 내 query로 전달된다고 해석하면 안 된다. 역할·shell attention readout은 모든 valid hidden node를 직접 읽는 별도 경로다. 연결성·readout 존재만으로 48/96/192 중 충분한 노드 수, 최적 그래프, 학습 성능을 선언하지 않는다.

## 비용 측정과 남은 병목

같은 사전 결속 physical 32 tile에서 각 경로별 disposable warmup 1회 후 model/buffer/RNG를 복원하고, fresh AdamW의 **연속 3 update**를 측정했다. CNN→selection→graph→L1/L2→전체 loss→backward→finite/clip→optimizer가 측정 범위다. loader 대기, 실제 production checkpoint 저장, 진단용 parameter delta CPU 복사는 제외된다. 같은 tile의 짧은 update이며 epoch throughput 측정이 아니다.

| branch seed | 이전 update 중앙값(s) | relay update 중앙값(s) | 새 update mean(s) | post-update 전체 DEBUG memory/teacher refresh 이전→새(s) |
|---|---:|---:|---:|---:|
| 48 | 0.561204 | 0.602663 | 0.599077 | 0.326389 → 0.414202 |
| 96 | 0.767630 | 0.810237 | 0.807037 | 0.446354 → 0.536821 |
| 192 | 1.256679 | 1.202110 | 1.193508 | 0.746752 → 0.796024 |

새 relay 경로는 48/96에서 약 7.4%/5.6% 느려졌고 192에서 약 4.3% 빨라졌다. 연결 복구에 드는 비용과 동일 donor scene 재사용 이득이 함께 들어간 실제 경로 비교이며, 연결성 수정이 모든 밀도에서 속도를 높였다고 보고하지 않는다. 초기 첫 baseline memory refresh의 39.94초 raw cold loading을 새 경로 compute 시간과 비교하지 않는다.

CUDA event로 분리한 새 forward 평균은 다음과 같다. 모든 행은 동일 3회 update의 평균이며 이 합에는 L1/L2·loss와 진단 경계 밖의 비용이 모두 포함되는 것은 아니다.

| seed | CNN(s) | FPS seed selection(s) | relay retention(s) | selection total(s) | graph/proof(s) | SAGE 3층(s) |
|---|---:|---:|---:|---:|---:|---:|
| 48 | 0.011483 | 0.145474 | 0.077881 | 0.229303 | 0.047960 | 0.004339 |
| 96 | 0.012043 | 0.294081 | 0.089235 | 0.389353 | 0.079291 | 0.007721 |
| 192 | 0.011944 | 0.567892 | 0.084275 | 0.658095 | 0.175734 | 0.013551 |

대표점 FPS 선택이 여전히 큰 비용이다. graph 구성의 `[B,N,N]` 거리 계산과 관계별 sorting도 남아 있어 모든 연산이 sparse-linear time인 구현은 아니다. 현재 작은 message graph의 3층 연산만 빨랐다는 결과를 3시간/epoch 해결로 환산하지 않는다. support refresh·validation·loader/save를 포함한 서버 epoch와 전체 학습 성능은 미측정이다.

## 자원·검증·시각화

실제 RTX 5070 Ti 1개, VRAM 15.920 GiB(시작 가용 14.659 GiB), CPU 8 physical/16 logical, 시작 가용 RAM 약 39.373 GiB를 확인했다. 명시적인 CUDA 12 GiB/RSS 32 GiB/resident 12 GiB 한도, parallel raw reader 4개, cached native crops, pinned-memory/non-blocking 전송을 사용했다. 비교 update 입력은 `[33,1,58,50,7]`이고 unique CNN crop 33개와 physical pair 32개를 유지했다. FP32, TF32 off, autocast off, deterministic 설정이다.

전체-case coverage 단계 peak allocation은 4.585 GiB였다. 새 48/96/192 update peak는 각각 약 1.780/1.780/1.928 GiB이며 여섯 model clone을 보유한 동일 프로세스 측정이다. 이 수치를 단일 production model의 최대규모 VRAM admission으로 사용하지 않는다. 최종 RSS 약 5.994 GiB, 전체 133관측 forward/coverage/export는 42.811초였다.

- 구현/회귀 검사 41개 PASS: numerical CUDA 40개와 순수 profile metadata 1개. 원본 seed 보존, nearest-three가 잘라낸 Z 연결, concave organ hole, unreachable seed 보존, invalid input, falsified parent/path, relation degree/SpMM, dynamic-role coverage, physical32 exact scene 재사용과 independent-expanded selection parity를 포함한다.
- 실제 CT 여섯 경로에서 각 3회 연속 full update. CNN·SAGE·readout/fusion·L1·L2의 실제 finite/nonzero gradient, optimizer 포함 및 parameter delta를 확인했다. 새로운 graph selection의 이산 index 자체는 미분 가능하지 않지만 선택된 CNN feature와 사용되는 모델 경로는 loss에 연결된다.
- 저장된 native graph의 독립 metadata/BFS/path 감사와 관련 순수 JSON unit 5개 PASS. 두 모델×세 밀도×P5/U3＝48개 표시 graph에서 원본 seed/query 좌표와 역할 보존, CUDA의 component/isolate counts 일치를 확인했다. 24개 새 표시 graph는 모두 component 1, isolates 0이며 양쪽 query의 weak component 밖 노드도 0개다. 23,650개 mandatory native polyline의 좌표·한 step·bbox·누적 길이를 독립 확인했고 최대 누적 길이는 5.712891 mm였다. 이 감사는 CPU로 학습 모델을 실행하지 않으며 CT/mask pixel을 읽지 않는다. mask support는 앞선 실제 CUDA validator/exporter가 검증한 기록이고 이 JSON 감사가 독립 재검증한 것은 아니다. 전체 133개의 수치는 GPU report 집계이며 independent BFS는 표시 48개에 한정된다. `independent_connectivity.json`에 FP64 native path의 고정 1e−5 mm tolerance와 FP32 node 좌표의 최대 1.525879e−5 mm serialization quantization 차이를 별도로 기록했다. tolerance를 자동 완화하지 않았다.
- 실제 HTML browser 검사 27개 PASS. 세 밀도와 이전/relay 모델, P 5개/U 3개 표시, native-mm 공통 축척·회전·zoom·좌표 선택·typed relation 필터·종양 overlay·mobile overflow·browser error를 확인했다. 확대 배율과 방향 관계 필터의 edge 수를 실제 export와 대조한 추가 검사도 통과했다. 최종 보존 기록은 `validation/relay_sparse_20261002/verification_final.json`이다.

3D 시각화는 모든 133관측 중 실제 P 5개와 U 3개를 선택해 표시한다. 삼각형은 추가 relay, 원/사각형은 원래 seed, query 표식은 원래 query다. mandatory edge는 실제 parent native voxel 경로의 polyline으로 그린다. 일반 message edge는 관계 edge이며 혈관이나 해부학적 구조라는 주장을 하지 않는다. 모든 원본 native/mm 좌표·edge·anatomy를 유지한 lossless path dictionary 압축으로 fragment 723,036 bytes를 만들었다. displayed 8개 graph의 시각화 범위와 전체 133개의 GPU 검증 범위를 구분한다.

로컬 fragment: `C:/Users/user/.codex/visualizations/2026/09/17/01a0ae06-ab24-7fe0-bdb8-ef2751a1c7be/v1-relay-sparse-ct.html`.

## 재실행과 결과 위치

다음은 동일 실제 CT DEBUG probe 명령이다. 출력 폴더는 기존 결과가 없는 새 이름을 사용하며, production 학습과 checkpoint 생성을 시작하지 않는다.

```powershell
.venv/Scripts/python.exe -u tools/verify_relay_sparse_ct_debug.py `
  --run work/local_cnn_experiment_resume_DEBUG_20261001/resumed `
  --assignment work/v222_v1_full_training_20260924/cache/pair_assignment.json `
  --assignment-receipt validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json `
  --case liver_66 --output work/relay_sparse_CT_DEBUG_20261002_new `
  --query-radius-mm 3 --near-radius-mm 5 --mid-radius-mm 10 `
  --workers 4 --cuda-gib 12 --rss-gib 32 --resident-gib 12 `
  --coverage-workspace-mib 96 --warmup-updates 1 --measured-updates 3
```

새 단위 검사는 `.venv/Scripts/python.exe -m unittest discover -s tests -p test_relay_sparse_graph.py -v`로 실행한다. 기존 `test_relational_sparse_graph.py`, `test_sparse_feature_graph.py`, `test_sparse_feature_coverage.py`도 함께 회귀 검사했다.

- 실제 원시 GPU 결과: `work/relay_sparse_CT_DEBUG_20261002_r2/report.json`.
- 실제 native export: 같은 폴더의 `visual_payload.json`.
- Git 전달 수치: `validation/relay_sparse_20261002/actual_CT_DEBUG_summary.json`.
- 독립 graph/path 감사: `validation/relay_sparse_20261002/independent_connectivity.json`.
- 새 sampling/연결 구현: `l0_sparse_feature/relay_sampling.py`, `l0_sparse_feature/relayed.py`.
- 실제 probe/요약/시각화 생성기: `tools/verify_relay_sparse_ct_debug.py`, `tools/summarize_relay_sparse_debug.py`, `tools/render_relay_sparse_debug.py`.

이전 `docs/relational_sparse_L0_DEBUG_20261002.md`에 기록된 분리된 그래프 결과는 역사적 결과로 보존한다. 이번 경로는 해당 연결 문제를 고쳤지만 **전체 학습·전체 평가·추천 정확도·epoch 단축·production-ready 검증은 완료하지 않았다.** 장기 GNN/nnU-Net 학습이나 production checkpoint/ready 표시를 만들지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 최종 실행은 OOM 없이 완료했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
