# v2.2 고정 입력 재사용 구현과 남은 시간 병목

현재 3시간/epoch 문제는 해결되지 않았다. Basic CP, No-CP, 모델·그래프·후보·support 수,
physical batch, epoch, CP80%, seed42는 변경하지 않았다. 전체 학습을 시작하지 않았다.

## 구현한 실행 후보

`tools/v22_static_reuse.py`는 다음 데이터를 재사용한다.

- 같은 support episode의 L1 data–label 연결, 관계 입력, L2 동일 task 차단 mask.
- 같은 query/label shape의 연결 인덱스와 U 관계 입력. 학습된 edge projection은 재계산한다.
- 같은 CT의 비교 관측 목록, epoch memory에서 가져온 고정 reference embedding.
  현재 query의 L0 출력은 원래처럼 index_copy로 넣고 gradient를 유지한다.
- checkpoint의 고정 memory·memory_generation·plan·plan_generation CPU snapshot과 hash.
  현재 모델 및 buffer hash, RNG hash, Adam 저장, 매 batch 원자적 저장/fsync/완료 receipt는 유지한다.
  자라는 memory_work prefix는 매번 새로 복사/hash한다.

L1/L2의 학습 출력, attention, dropout, loss, gradient, optimizer update는 캐시하지 않는다.
CPU/GPU tensor 교체 또는 정상적인 in-place 변경의 version, shape, stride, device와 metadata
변경 시 캐시를 갱신한다. .data나 외부 NumPy로 storage를 몰래 변경하는 사용법은 허용하지 않는다.
메모리·plan 변경을 놓치지 않도록 mutation 검사를 추가했다. 캐시는 마지막 episode/case만
유지하며, checkpoint writer가 완료되기 전 snapshot을 갱신해서 덮어쓰지 않는다.

원래 `tools/v222_support_snapshot.py`는 이미 support 갱신 중 모델·Adam CPU 복사를
재사용하고 있었다. 그 기능을 새 개선처럼 주장하지 않는다. 기존처럼 고정 pass 중 모델이
바뀌면 오류를 내며, 새 코드는 optimization 중 고정 memory/plan도 재사용한다.

## 실제 확인한 것

- 짧은 검사 31개 통과(4.128초): 새 코드, 기존 CUDA checkpoint parity, snapshot 및 resume
  무결성 회귀 검사. 초기 저장 테스트는 sandbox 임시 경로 권한 오류였고, 작업 폴더의 테스트
  전용 임시 디렉터리로 정상 재검사했다.
- CUDA 합성 support **11,279개**, hidden128/head4 및 L1/L2 기존 깊이를 유지한 두 optimizer
  update에서 loss, pre-clip 전체 gradient, 모델, Adam, RNG bitwise 일치. 평가 출력도 일치.
  이 검사의 L0는 명시적인 linear fixture다. 실제 CT L0/full pipeline/MIG 검증은 아니다.
- snapshot 재사용 후 원본 seal 함수로 다시 계산한 현재 모델/RNG/memory/plan hash가 모두 일치.
  실제 파일 저장과 durable step, model/buffer 변경, prefix 증가, memory mutation도 확인했다.
- RTX5070Ti 부분 측정에서 warm-up 제외 L1/L2+linear fixture 평균은 원본 **0.0818904초**,
  후보 **0.0811137초**. 각 3회뿐이고 범위가 겹친다. 의미 있는 속도 개선으로 판정하지 않는다.
  원자료: `work/v22_static_reuse_component_20260928_DEBUG.json`.
- 실제 서버 제공 checkpoint 비용은 약0.271초/22.104초 step였다. 이를 전부 제거한다고 가정해도
  약1.23%다. 저장 최적화만으로 며칠 걸리는 문제를 해결할 수 없다.

## 짧은 실제 update 비교에 연결

production entry point는 변경하지 않았다. core/cache/runtime identity도 유지한다.
`benchmark_v22_recompute_debug.py --candidates static_reuse`가 실제 원본 checkpoint의 다음
full batch들과 전체 support로 이 후보를 호출한다. source SHA를 기록하고 원본 checkpoint는
읽기만 한다. 새로운 checkpoint로 원본을 재라벨링하지 않는다.

`--checkpoint-cost`를 추가하면 원본 saver와 ReuseSaver의 snapshot·요청·디스크 직렬화·완료까지
별도 측정한다. 이 파일은 `debug=True`, `artifact_kind=debug_execution_snapshot`이며 production
resume에 사용할 수 없다. CPU parity 복사는 계산 시간에서 제외한다. 저장은 매회 flush하므로
실제 비동기 학습의 overlap을 반영한 전체 step/epoch 시간이라고 해석하지 않는다. 각각의
단독 비용을 확인하기 위한 검사다. 정책 종료 시 해당 saver의 GPU 참조도 해제한다.

현재 학습 화면에서 Ctrl+C 후 PAUSED가 확인된 상태에서만 실행한다.

```bash
conda activate nnunet &&
cd /home/aicompetition06/Medical/HierCP-v22-e1e34bf &&
git fetch origin codex/v222-server-r6 &&
git switch --detach FETCH_HEAD &&
python -u tools/benchmark_v22_recompute_debug.py \
  --cache work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --checkpoint work/v22_gnn40_resume114_20260928/training/checkpoint_latest.pt \
  --output work/v22_static_reuse_20260928_DEBUG \
  --candidates static_reuse --operators --checkpoint-cost
```

기본 3batch는 각각 warm-up1+측정2이며 원본/후보 합계6update다. 학습 규모를 줄이는 설정이나
새 성능 실험이 아니다. 실제 서버 수치가 없으므로 이 명령으로 어느 정도 단축될지 보장하지 않는다.
이 비교에서도 빨라지지 않으면 production으로 승격하지 않는다.

## 미완료 항목과 판단 기준

L0 전체 outer checkpoint 해제 후보는 이전 MIG 검사에서 OOM으로 탈락했다. 한 block씩 해제하는
별도 후보는 아직 실제 서버 결과가 없다. 이번 static_reuse는 그 checkpoint 정책을 변경하지 않는다.
현재 전처리 재사용만으로 큰 단축이 입증되지 않았으므로 L0 연산·재계산 병목은 남아 있다.

실제 완료 판정은 같은 전체 작업량의 update, support refresh, validation, 저장/대기를 합친
epoch 비용 감소다. 소수 batch를 전체 epoch 실측으로 바꾸어 보고하지 않는다. 기존 runtime_events의
완료된 epoch 기록과 이후 동일 범위 기록을 사용해야 한다. 본학습 재시작, nnUNet 추가 실험,
G3/G4 통과 또는 전체 최적화 완료를 이번 작업으로 선언하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 작은 검사는 명시적 DEBUG다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 서버 비교는 saved batch를 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. RTX5070Ti16GB, 조회 당시930MiB/0%.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 검사를 실제 CT로 보고하지 않는다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. CUDA 동일성 검사를 수행했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
