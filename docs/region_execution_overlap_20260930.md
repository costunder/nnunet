# v2.2 저장 겹침 실행·검증된 GPU 캐시·SAGE 작업 공간

사용자 요청: 반복 병목을 실제 검사해 개선하고, 안전한 한도 안에서 VRAM을 적극 활용한다. 이번 변경은 `10287dd`의 환자 support episode 학습을 보존하는 실행 경로 변경이다. 모델·loss·support16·전체 관측·query batch·후보128·원본 paste mask를 변경하지 않는다.

## 확인한 비용과 수정

| 기존 경로 | 수정 |
|---|---|
| 매 update 모델·Adam·memory를 CPU로 복사한 뒤 hash·fsync 완료까지 기다림 | GPU dtype/device별 묶음 복사로 독립 snapshot 생성. hash·atomic 저장은 단일 writer에서 다음 update와 겹침 |
| epoch 동안 고정된 memory와 episode plan, best를 반복 D2H 복사 | tensor identity/version을 확인한 독립 CPU snapshot 재사용. 변경되면 다시 복사 |
| memory refresh의 이전 완료 chunk까지 반복 D2H 복사 | 완료 chunk별 CPU snapshot 재사용. 새 chunk만 복사; 전체 payload hash 검증은 유지 |
| parameter별 유한값 검사마다 GPU→CPU 판정 | 모든 gradient의 유한값을 GPU에서 검사하고, 정상 경로에서는 한 번의 CPU 판정. 실패 시 이름별 상세 오류 |
| 반복 batch의 H2D 및 같은 구조의 전체 GPU 검사 | 최초에 원래의 전체 검사 수행. 이후 CPU/GPU tensor identity/version 확인을 거쳐 검증된 GPU batch 재사용 |
| 같은 그래프의 CSR·전치행렬을 batch 전환 후 다시 생성 | 검증된 batch와 함께 VRAM 캐시에 보존. 재사용 시 변조 확인 및 기존 edge identity 검사 유지 |
| SAGE edge 작업 공간 고정 64MiB | 명시적 `--sage-workspace-mib` 사용. row/edge를 버리지 않고 한 번에 처리할 row 묶음만 확대. backward도 forward의 같은 한도 사용 |

GPU cache는 LRU이며 **학습 sample을 제거하지 않는다**. 캐시보다 큰 batch는 원래 batch 전체로 실행하고 상주 재사용만 하지 않는다. 캐시에 입력 tensor와 CSR/transpose 저장량을 함께 계산한다. 모델 activation은 기존 retained 모드다. 미검증 partition의 research-report 정책과 데이터 무결성 검사는 유지한다.

## 저장과 재개의 안전성

- 학습 중인 GPU/CPU tensor를 writer에 직접 넘기지 않는다. 모든 동적 model/Adam/RNG/cursor는 독립 snapshot이다.
- writer는 한 개이며 이전 저장 오류를 다음 저장 또는 종료에서 반드시 전달한다. 무한 queue·save 생략 없음.
- 저장 횟수는 기존처럼 매 update. 이전 정상 checkpoint는 새 파일 flush/fsync 후 atomic replace될 때까지 유지.
- Ctrl+C에서는 현재 batch 완료와 모든 저장 완료를 기다린 후에만 PAUSED 출력.
- phase 전환·정상 종료에서도 마지막 쓰기까지 기다린다. 오류 종료 시에도 이미 제출한 정상 snapshot 쓰기를 정리하고 오류를 전달한다.
- snapshot의 tensor 값뿐 아니라 **channels-last-3d 메모리 형식**을 보존한다. 초기 실제 재개 검사에서 fused Adam layout 오류를 발견해 수정했고, 회귀 검사와 새 프로세스 재개를 다시 수행했다. 실패한 초기 DEBUG 결과는 삭제하지 않았다.
- `--resume-execution-upgrade`는 검토된 `10287dd`의 학습 상태 또는 같은 소스의 명시적 실행정책 변경만 수용한다. 모델/Adam/RNG/memory/plan/epoch/step/query cursor/physical batch를 보존하며 support 학습정책을 바꾸지 않는다.

## VRAM 설정

A6000 서버 예시: 총 PyTorch CUDA 한도40GiB, 그 안에 GPU 입력/CSR cache 최대8GiB, SAGE workspace512MiB. 캐시8GiB가 총 한도에 더해지는 것이 아니다. workspace는 연산 중 임시 tensor 크기를 정하는 값이며, gather와 multiplication 중간 결과 등 때문에 전체 peak가 정확히512MiB인 것은 아니다. 총40GiB 제한과 실제 peak 측정은 계속 적용된다.

숫자를 채우기 위해 VRAM을 미리 할당하지 않는다. 반복 입력·희소 구조 저장과 더 큰 작업 묶음에만 사용한다. 최초 batch들은 cache miss이며, 작은 실제 입력이면 8GiB를 전부 사용하지 않는다. 물리 batch는 저장본의 값을 유지한다. 이 변경 후 서버 batch32/48/64의 처리량 우열을 새로 측정한 것은 아니다.

## 검증 범위

최종 원문은 `validation/region_execution_overlap_20260930/`에 보존한다.

최종 단독 GPU 실행(5070 Ti, torch2.8.0)의 동일 실제 CT DEBUG 4 updates 비교:

| 범위 | 기존 동기 실행 | 수정 실행 |
|---|---:|---:|
| optimization wall time, 마지막 저장 완료 포함 | 4.295초 | 3.551초 |
| 평균 update, 마지막 drain 제외 | 0.938초 | 0.706초 |
| update 내 snapshot·이전 저장 대기 | 0.338초 | 0.047초 |
| DEBUG memory refresh, 마지막 저장 완료 포함 | 1.093초 | 0.881초 |
| DEBUG validation | 0.366초 | 0.348초 |

optimization 구간 감소는 17.3%다. 작은 DEBUG 1회 비교이며 서버 batch32/support16의 epoch 개선율이 아니다. 실제 최종 model/Adam/state/RNG는 동기/수정 및 중단/재개 비교 모두 정확히 일치했다. GPU batch 캐시의 별도 cold/hit 측정은 0.168초/0.00134초였으며, 실제 학습 hit율은 epoch·batch 재사용 여부에 달린다. workspace 확대는 출력/gradient 일치까지 검증했지만 서버 처리량 개선은 아직 측정하지 않았다.

- snapshot 독립성·순서·저장 오류 전달·정적 복사 재사용·gradient 검사·channels-last 형식·작업 공간 CPU/GPU 출력 및 gradient 일치 검사.
- 실제 CT DEBUG train8/val2, 실제 query batch2. synchronous/overlapped의 최종 model/Adam/state/RNG 비교.
- 새 프로세스 pause/resume 일치, `10287dd` checkpoint에서 실행 업그레이드 후 다음 step 진행.
- 검증된 GPU 입력 cache hit 및 변조 거부. 다른 batch를 처리한 후 기존 CSR/transpose 복원 시 rebuild 없이 출력/gradient 일치.
- 학습, DEBUG 전체 memory refresh, validation, best 선택, final memory 및 export 경로 확인.
- 처음 full comparison 중 별도 짧은 CUDA 단위 검사와 겹친 측정은 속도 결론에 사용하지 않는다. 별도 GPU 작업 없이 다시 수행한 최종 검사 원문을 기준으로 삼는다.

`update_timing.jsonl`은 support 준비 wall time, H2D·검사, forward/backward, gradient/optimizer, snapshot 및 이전 writer 대기, cache bytes/hit/miss/peak를 기록한다. `checkpoint_timing.jsonl`은 실제 hash와 write/fsync를 별도로 기록한다. `phase_timing.jsonl`의 optimization 시간은 **마지막 checkpoint flush까지 포함**하며 memory refresh와 validation도 별도 기록한다. async enqueue 시간만을 전체 저장 시간으로 표시하지 않는다.

로컬 소규모 DEBUG 비용은 서버 epoch 예상치가 아니다. A6000 전체 그래프·support16의 실제 시간, 서버 NFS 지연과 최악 입력 peak는 아직 확인하지 않았다. 정상 forward/backward, 새로운 환자 episode의 군집 생성, 최초 입력검사, 필수 저장 비용은 남는다. GPU가 항상100%이거나 모든 병목이 사라졌다고 보장하지 않는다.

## 서버 적용

기존 학습 창에서 Ctrl+C 한 번 후 PAUSED와 RESULT를 확인한다. 서버 프로세스나 SSH 세션을 종료하는 명령은 사용하지 않는다. 아래 `read`에 **방금 저장된 RESULT 경로**를 입력한다. 과거 step73/173으로 고정하지 않는다.

```bash
cd /home/aicompetition06/Medical/HierCP-regions-8580e59 &&
read -r -p "방금 저장된 checkpoint_latest.pt 경로: " CP_RESUME &&
test -f "$CP_RESUME" &&
git fetch origin codex/v222-server-r6 &&
git switch --detach FETCH_HEAD &&
CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9 \
python -u tools/run_fixed_regions.py train \
  --profile-policy research-report \
  --cache work/regions_frozen_reuse_20260929_230052/cache/index.json \
  --resume "$CP_RESUME" --resume-execution-upgrade \
  --support-patients 16 --activation-storage retained \
  --execution-pipeline overlapped --device-cache-gib 8 \
  --sage-workspace-mib 512 \
  --output "work/regions_overlap_$(date +%Y%m%d_%H%M%S)" \
  --workers 16 --cuda-gib 40 --rss-gib 192 --resident-gib 128 \
  --batch-candidates 32 48 64
```

이 변경 이후 같은 실행 설정으로 다시 재개할 때는 일반 `--resume`만 사용한다. `--resume-support-minibatch`를 다시 넣지 않는다. prepare·초기 support 재생성 없이 저장된 진행 위치에서 이어간다. 이번 로컬 작업에서는 서버 장기 학습을 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 명시적 VRAM cache 한도는 데이터 한도가 아니다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 batch 보존 및 저장 overlap 적용.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 GPU 검사에서 모델 축소 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사는 실제 CT와 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
