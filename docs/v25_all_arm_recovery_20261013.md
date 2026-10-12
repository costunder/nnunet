# v2.5 GPU1·4·5·6 전체 재개 및 배포 사고 기록

이 문서는 2026-10-13 KST의 전체 재개 기록이다. 같은 날 앞선 `v25_gpu456_terminal_recovery_20261013.md`의 관찰 뒤 GPU1/C가 추가 중단됐으며, C 복구 배포 실수로 A/B/D도 중단됐다. 이전 관찰의 성공을 현재 상태로 사용하지 않는다.

## 중단 원인

1. GPU1/C는 종료 중인 worker의 status와 빈 argv를 서로 다른 순간에 읽는 identity 검사 경계에서 중단됐다. 새 실행 경로는 동일 PID·생성 시각·UID를 제한된 횟수로 재관찰하고 실제 zombie만 활성 자원 계산에서 제외한다. PID 재사용·argv 변조·권한 오류·살아 있는 미확인 프로세스는 계속 오류로 처리한다.
2. C-only 코드 배포가 `os.link`로 실행 중인 원본 코드를 연결했다. 내용 SHA는 같았지만 inode의 ctime·link 수가 바뀌어 A/B/D의 정상적인 입력 무결성 검사가 실패했다. 이는 배포 구현 오류다. 신호로 프로세스를 종료한 것이 아니며, ctime 검사를 완화하지 않았다.
3. 기존 GPU4 U128 forward의 RSS pressure 처리와 worker 종료 경계 수정도 새 별도 복사본에 유지한다. 비활성 cache·allocator 회수 후 원래 RSS64GiB·CUDA40GiB 검사를 실행한다. 활성 입력·모델·그래프·B4·workers4를 줄이지 않는다.

## 새 배포 및 재개

배포 commit은 `f1c79f13a5ca9f38c7ad73691fee57ddb6799e7b`이며 승인된 `codex/v24-cpu-pipeline` 브랜치에 push했다. 원본에서 별도 inode의 새 파일로 복사하고 원본 dev·inode·mode·UID·GID·size·mtime·ctime·link 수가 전후 동일함을 검증했다. start 전에도 이 메타데이터와 내용 SHA를 다시 확인했다. 새 배포는 hardlink를 만들지 않았다.

새 코드 경로는 `/var/tmp/aicompetition06_v25_20261012/run_1791784719489478600/code_ALL_recovery_f1c79f13a5ca`다. 단일 새 controller PID2576612, birth1791820676.24, UID41833 아래 네 arm의 독립 chain을 동시에 실행했다. 모든 이전 controller와 네 이전 GNN actor가 자연 종료됐음을 확인한 뒤 실행했다. GPU0의 nnU-Net actor PID2460965, birth1791804951.51, UID41833은 유지했다. 어떤 프로세스에도 신호를 보내지 않았다.

| GPU / arm | 새 GNN PID | 복구 checkpoint | 실제 시작 단계 / 환자 cursor | U |
|---|---:|---|---|---:|
| GPU6 / A | 2576933 | e6 / update102 | train_probe 20/65 | 16 |
| GPU5 / B | 2576934 | e7 / update119 | train_probe 0/65 | 32 |
| GPU1 / C | 2576931 | e5 / update68 | training 0/65 | 64 |
| GPU4 / D | 2576932 | e4 / update68 | stage_validation 0/21 | 128 |

이미 학습65명이 완료된 A/B/D는 그 epoch의 학습을 다시 수행하지 않고 저장된 evaluation cursor부터 이어 간다. C는 저장된 training cursor부터 이어 간다. 자기 optimizer·scheduler·AMP scaler·global/shuffle RNG·curriculum·환자 cursor를 포함한 전체 checkpoint를 사용한다.

GNN40epoch/680updates, physical B4, workers4, arm별 기존6CPU affinity, RSS64GiB/CUDA40GiB와 전체416GiB RAM guard를 유지했다. A=U8/+8, B=U16/+16, C=U32/+32, D=처음부터U128이며 adaptive 증가 기준도 원 계약 그대로다. 후보128 전체 cache와 실제 runtime의 v1 archive를 재사용했다. 모델·loss·context·분할·전체 데이터 및 scientific source23개는 내용 SHA까지 유지했다.

각 chain은 자기 full128 GNN BEST → 전체 CP642×128 → nnU-Net250epoch → 자기 `checkpoint_best.pth` 전체26명 평가로 연결한다. 단일 arm 실패가 다른 arm이나 GPU0에 종료 신호를 보내지 않는다.

## 저장 및 뷰어

원 canonical FAILED monitor와 원 로그는 SHA로 봉인해 보존한다. 새 monitor는 각 arm의 `monitor/ALL_recovery_f1c79f13a5ca/status.json`이다. 새 `view_v25_current.py`는 기존 UI2 렌더러를 그대로 사용하고 이 새 monitor만 읽는다. 원 `view_v25.py`는 이전 FAILED 실행 기록을 읽는다.

디스크의 원 latest/BEST checkpoint는 유지한다. 시작 당시 checkpoint4개 사본은 별도 tmpfs에 두었으며 영구 백업으로 주장하지 않는다. 전체 native 출력·tmpfs reserve·기존 GPU0의 디스크1GiB 및 tmpfs 실행 예산을 포함해 저장공간을 검사했다. 실행 중 사용하지 않는 과거 문서·archive만 새 코드 복사 대상에서 제외했고, 실제 v1 archive·runtime·config·23개 scientific source를 모두 복사했다. 기존 파일 삭제·관리자 권한·새 mount를 사용하지 않았다.

U24 등 신규 curriculum 단계의 UI `U 후보 그래프 준비`는 SHA 검증·기존 U128에서 정확한 record-ID/query 순서로 선택·새 active 후보 집합의 원 KNN/prototype CPU geometry 재계산·새 immutable prefix cache 저장을 포함한다. 4 workers를 사용하며 CT decode·EDT·L0 forward를 이 준비 단계에서 다시 실행하지 않는다. 원 full128 cache가 없으면 명시적 오류이며 cold raw builder fallback은 없다. 한 번 만들어진 동일 prefix는 다음 epoch에서 검증 후 재사용한다. 학습 중 L0 weight는 실제 optimizer로 갱신되므로 neural feature를 이전 weight로 cache해서 재사용하지 않는다. 같은 epoch/model/RNG/21명인 U128 stage/full validation만 실제 결과를 재사용해 중복 L0/GNN forward를 제거한다.

신규 U24 준비 중 RSS 약50GiB에서 logical eviction0인 upper GC/heap trim 로그가 반복됐다. 이는 추가 성능 조사 대상으로 남긴다. 각 GC의 wall time 기여를 profiler로 측정하지 않았으므로 느린 준비 시간 전체를 GC 때문이라고 단정하지 않는다. 실행 중 봉인된 runtime을 수정하거나 RAM guard를 완화하지 않았다.

## 실제 검증

- 로컬 CPU 회귀116개 통과. 새 배포 묶음의 실제 서버 nnunet 환경 CPU 회귀83개, 6.266초, 통과. 숫자가 다른 이유는 로컬 실행에 기존 standalone AD33개가 추가됐기 때문이다.
- 재개 후 public viewer A/B/C/D 모두 returncode0, UI2, RUNNING, 오류 없음.
- 독립 원본 감사 `V25_FULL4_INDEPENDENT_PROVENANCE_AUDIT_READONLY_1791821274704313500.json`: 원 source954개 파일의 metadata 변화0개, 원본과 새 code가 같은 inode인 파일0개. 실제 U128·ABD 이전 seal 모두23개 scientific SHA가 새 FULL4와 일치하고 v1 archive 전체가 보존됐다. 백업4개648,527,304bytes SHA 한 pass는 실제0.421초였다. 원래30초 resource poll을 유지하고 중복 guard hash 최적화는 이번 runtime에 적용하지 않았다.
- CPU로 atomic checkpoint의 하나의 열린 file descriptor를 고정해 읽고 tensor-value checksum, 원 실험 identity, 전체 AdamW state537개, 실제 optimizer step, scheduler40epoch, RNG, deterministic curriculum/cursor를 검증했다. 읽기 검사에서 CUDA는 초기화되지 않았다.
- `V25_ALL_OPTIMIZER_SNAPSHOT_AUDIT_READONLY_1791820900333326200.json`: A e6 stage_validation4/21, B e7 train_probe36/65, C e5 train16/65 및 update72(복구 전68), D e4 stage_validation8/21. 네 arm 모두 실제 cursor 또는 optimizer step이 증가했다. A/B/D의 평가 중 update가 유지되는 것은 정상이다.
- `V25_ALL_CURRENT_READONLY_1791821250356149500.json`: D가 e4 validation을 완료하고 e5의 실제 B4/U128 학습8/65, update70으로 진행했다. C도 e5 학습65/65와 update85를 완료해 train_probe로 전환했다. A e6 full128 validation18/21, B e7 full128 validation8/21. D의 비활성 RAM 회수는 실제 로그에서 약33.7GB RSS 감소를 기록했다.
- `V25_ALL_CURRENT_READONLY_1791821932586985900.json`: A e7 U24 준비42/86, B e8 U32 학습8/65와 update121, C e5 full128 validation4/21, D e5 U128 학습62/65와 update84. C의 실제 종료 worker PID2584750/birth1791821621.68/UID41833을 zombie로 기록하고 신호 없이 제외한 뒤 다음 validation으로 진행했다. C stage validation 마지막 batch 중 관찰한 일시 RSS67.33GiB는 회수 이후56.78GiB로 낮아졌으며, 원래 메모리 검사를 완화하지 않았다.
- `V25_ALL_RECOVERY_STAGE_1791820633643907700.json`과 `V25_ALL_RECOVERY_START_1791820676501137700.json`에 원본 메타데이터·실제 자원·복사/시작 증거가 있다. 로컬 receipt들은 `outputs/v24_gpu6_recovery_20261011`에 저장했다.

최종 복구 확인은 `V25_ALL_CONTINUATION_PROGRESS_VERIFIED_READONLY_1791822806356863000.json`이다. 네 arm 모두 같은 새 GNN PID가 RUNNING이고, 자기 복구 전보다 optimizer step이 증가했으며, 다음 epoch에 도달한 것을 동시에 검증했다. 각 실제 AdamW parameter step의 min/max가 아래 update와 같고 tensor checksum·identity·scheduler·RNG·curriculum 검증도 통과했다. 원 FAILED monitor·봉인된 source SHA와 GPU0의 원 PID/birth/UID/argv도 보존됐다.

| GPU / arm | 실제 epoch | 학습 U | 복구 전 → 실제 optimizer update | 현재 단계 |
|---|---:|---:|---:|---|
| GPU1 / C | 6/40 | 64 | 68 → 90 | 학습20/65 |
| GPU4 / D | 5/40 | 128 | 68 → 85 | train 범위 평가36/65 |
| GPU5 / B | 8/40 | 32 | 119 → 136 | train 범위 평가 입력 준비 |
| GPU6 / A | 7/40 | 24 | 102 → 110 | 학습32/65 |

현재 뷰어는 다음 명령으로 실행한다. 같은 UI2를 유지하며 각 arm의 새 monitor를 읽는다.

```bash
V25_PY=/home/aicompetition06/.conda/envs/nnunet/bin/python3.10
V25_VIEW=/var/tmp/aicompetition06_v25_20261012/run_1791784719489478600/view_v25_current.py
"$V25_PY" -B "$V25_VIEW" --arm C # GPU1
"$V25_PY" -B "$V25_VIEW" --arm D # GPU4
"$V25_PY" -B "$V25_VIEW" --arm B # GPU5
"$V25_PY" -B "$V25_VIEW" --arm A # GPU6
```

이는 실제 데이터 CUDA 학습·검증의 재개 관찰이다. GNN40epoch, 후속 nnU-Net250epoch 또는 전체26명 최종 평가 완료를 의미하지 않는다. 이후 전체 학습에 오류가 없다는 보장으로 사용하지 않는다.

재발 방지를 위해 과거 C-only 배포 도구도 별도 commit `e9f36b3`에서 독립 복사·원본 메타데이터 검증으로 수정했다. 해당 별도 수정은 C 관련 CPU 회귀123개를 통과했으며 현재 서버의 봉인된 FULL4 runtime에 덮어쓰지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 앞선 hardlink의 원본 메타데이터 변경 사고는 위에 명시했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 원 B4, 네 arm 동시 실행을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 optimizer 상태 증가를 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
