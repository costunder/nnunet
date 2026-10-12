# v2.5 GPU4·5·6 중단 복구 기록 — 2026-10-13

GPU4/D, GPU5/B, GPU6/A를 각자의 checkpoint_latest.pt에서 재개했다. 새 프로세스 생성뿐 아니라 저장된 optimizer step의 증가를 CPU로 읽어 확인했다. GPU1/C와 GPU0의 기존 작업은 같은 PID·생성 시각·UID로 계속 실행 중이다.

## 실제 중단 원인과 수정

- GPU5/B·GPU6/A: 종료 중인 worker의 status와 cmdline을 서로 다른 순간에 읽어, cmdline이 비어 있는 것을 살아 있는 잘못된 프로세스로 판단했다. 동일 PID·생성 시각·UID를 최대 3회 재관찰하고, 실제 zombie만 활성 자원 집계에서 제외한다. 여전히 살아 있는데 argv가 없거나 identity가 달라지면 오류를 유지한다. 프로세스에 신호를 보내지 않는다.
- GPU4/D: U128 forward의 RAM 검사에서 비활성 provider cache 및 allocator 회수가 제때 호출되지 않아 원래 64GiB RSS 제한을 넘었다. 실제 RSS가 기존 pressure target인 63GiB를 넘으면 기존 cache 회수 경로를 먼저 실행하고, 원래 64GiB RAM 및 40GiB CUDA 검사를 그대로 적용한다. 활성 tensor와 checkpoint payload를 버리지 않는다.
- 복구용 checkpoint 사본 3개는 별도 tmpfs 폴더에 저장했다. 원본 latest와 BEST는 기존 디스크에 남긴다. tmpfs 사본은 서버 재부팅 시 사라질 수 있으므로 영구 백업으로 간주하지 않는다. 기존 GPU0의 디스크 archive 1GiB, 전체 v2.5 저장 예산, tmpfs reserve 및 GPU0 tmpfs 실행 예산을 유지했다.

## 원래 실험 계약

| 항목 | 적용 값 |
|---|---|
| Python 환경 | `/home/aicompetition06/.conda/envs/nnunet/bin/python3.10` |
| GNN 목표 | 40 epoch, 680 optimizer update |
| Physical batch | 4; 기존 accumulation·parallel 설정 유지 |
| Worker | 4; arm별 기존 6 CPU affinity 유지 |
| GNN 자원 제한 | arm별 RSS 64GiB, CUDA 40GiB; 전체 기존 RAM guard 유지 |
| 후보 및 cache | 기존 전체 U128 cache 공유·재사용; CT 전처리 재실행 없음 |
| A / GPU6 | U8 시작, 기존 승인된 adaptive +8 정책 유지 |
| B / GPU5 | U16 시작, 기존 승인된 adaptive +16 정책 유지 |
| C / GPU1 | U32 시작, 기존 승인된 adaptive +32 정책 및 기존 프로세스 유지 |
| D / GPU4 | 처음부터 U128 |
| 후속 실행 | 각자의 full128 GNN BEST → CP 642×128 → nnU-Net 250 epoch → 각자의 checkpoint_best.pth 전체 26명 평가 |

모델, loss, 그래프, 데이터 분할, context 조건 및 frozen scientific source 23개는 변경하지 않았다. 복구는 optimizer, scheduler, AMP scaler, RNG, curriculum 및 환자 cursor를 포함한 전체 checkpoint로 수행했다.

## 재개 후 실제 checkpoint 검증

2026-10-13 00:12:57 KST부터 순서대로 읽은 atomic checkpoint snapshot. 파일 descriptor를 고정하여 checkpoint 교체 도중 서로 다른 파일을 섞어 읽지 않았다. CPU만 사용했고 CUDA 초기화는 전후 모두 false였다.

| GPU / arm | 재개 전 update | 실제 snapshot update | 실제 추가 update | epoch / 단계 | 저장된 환자 cursor | 실제 optimizer step |
|---|---:|---:|---:|---|---:|---:|
| GPU6 / A | 68 | 85 | 17 | e5 / stage_validation | train 65/65, validation 18/21 | 85 |
| GPU5 / B | 85 | 102 | 17 | e6 / train_probe | train 65/65, probe 40/65 | 102 |
| GPU4 / D | 51 | 57 | 6 | e4 / training | train 24/65 | 57 |

3개 모두 checkpoint identity SHA가 원래 자기 실험과 동일했고 optimizer state 537개가 유지됐다. 해당 snapshot의 RSS는 A 28.74GiB, B 19.03GiB, D 41.95GiB였다. GPU4는 재개 후 여러 실제 B4/U128 forward·backward·optimizer update를 완료했다. 이 관찰만으로 이후 40 epoch 전체에 오류가 없다고 보장하지 않는다.

00:18:56 KST의 후속 CPU checkpoint 관찰에서는 A가 e5의 학습·검증을 완료하고 e6으로 전환했고(update85), B는 e6 full128 validation 중(update102), D는 e4 학습 60/65 및 update66으로 진행했다. D의 실제 optimizer step도66이며 재개 전보다15회 증가했다. 동일 C·GPU0 actor와 새 ABD controller도 유지됐다.

00:20:33 KST부터 읽은 최종 CPU snapshot에서는 D도 e4 전체65명 학습을 마치고 train_probe로 전환했다. D의 실제 optimizer step68을 확인하여 세 arm 모두 이번 재개 후17회 update를 완료했다. A는 e6 준비, B는 e6 full128 validation16/21이었다.

## 배포 및 증거

- 배포 코드 commit: `2db93521a9462b2209a8f25c1c84e0bce5ba1855`; 승인된 `codex/v24-cpu-pipeline` 브랜치에 push 완료.
- 새 코드: `/var/tmp/aicompetition06_v25_20261012/run_1791784719489478600/code_ABD_recovery_2db93521a946`.
- 새 controller: PID 2553369, birth 1791817518.53, UID 41833.
- 새 GNN: GPU6/A PID 2553653, GPU5/B PID 2553654, GPU4/D PID 2553655.
- 보존된 GPU1/C: PID 2510833, birth 1791811642.45, UID 41833.
- 보존된 GPU0 native: PID 2460965, birth 1791804951.51, UID 41833.
- CPU 단위·복구 검사: 실제 서버 nnunet 환경에서 68 tests, 6.438s, OK.
- START receipt: `outputs/v24_gpu6_recovery_20261011/V25_ABD_RECOVERY_START_1791817518804460400.json`.
- 실제 update snapshot: `outputs/v24_gpu6_recovery_20261011/V25_RESUMED_UPDATES_READONLY_1791817978110705000.json`.
- LIVE receipt: `outputs/v24_gpu6_recovery_20261011/V25_RECOVERY_LIVE_READONLY_1791817887575219700.json`.
- 후속 실제 update snapshot: `outputs/v24_gpu6_recovery_20261011/V25_RESUMED_UPDATES_READONLY_1791818337098484300.json`.
- 최종 실제 update snapshot: `outputs/v24_gpu6_recovery_20261011/V25_RESUMED_UPDATES_READONLY_1791818434533537300.json`.
- 기존 뷰어 읽기 검증: `outputs/v24_gpu6_recovery_20261011/V25_VIEWERS_READONLY_1791818256109318600.json`; A·B·D returncode0, RUNNING, 오류 없음.

## 검증 범위와 남은 항목

실행 경로 수정, 서버 CPU 검사 및 실제 데이터의 CUDA 학습 재개를 확인했다. GNN 40 epoch, 후속 nnU-Net 250 epoch 및 최종 전체 26명 평가 완료를 이번 복구 결과로 주장하지 않는다. 새 후속 경로는 기존 정확한 완료 certificate 검증을 유지한다.

추가 읽기 전용 감사에서는 원 C controller가 종료 직후 잠시 zombie로 남는 경계에서 ABD lease가 보수적으로 오류를 낼 수 있는 경로를 확인했다. 실제 이번 장애는 아니다. 저장소에는 동일 controller identity와 실제 terminal C monitor/certificate를 검증하는 후속 수정을 준비했다. 실행 중인 SHA 봉인 코드를 바꿔 현재 작업을 중단하지 않았으므로 이 후속 수정은 현재 서버 runtime에 적용되지 않았다. 다음 안전한 배포가 남아 있다.

후속 종료 경계 수정은 로컬 standalone33개 및 기존 terminal 검사까지65개 테스트를 통과했다. 이는 현재 배포 코드의 서버68개 테스트와 별도 검증이다. PID 재사용, UID 변경, argv 변조, AccessDenied 및 가짜 zombie를 계속 거부하는 회귀도 포함한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
