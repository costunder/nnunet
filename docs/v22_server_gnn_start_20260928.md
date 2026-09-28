# 서버 캐시 준비 이후 GNN 학습 실행

## 상태와 실행 순서 정정

사용자가 제공한 서버 출력에서 `ced3eeb` 전체 캐시 생성(14,102 records)과 실제 CT GPU DEBUG 검사의 성공을 확인했다. 원격 파일을 직접 읽거나 이 로컬 작업에서 서버 학습을 실행한 것은 아니다.

GPU DEBUG 결과: 5,550,806 parameters, physical batch 32, support 6, finite gradients, ranking→L0 nonzero gradient, 다음 update의 loss/model/Adam 재개 일치. peak allocated 6,187,130,880 bytes. 이 수치는 full-support capacity나 steady-state throughput을 보장하지 않는다. 첫 update의 backward에는 추가 rank-gradient 검사도 포함된다.

G3 문서에는 학습 전 capacity 검사뿐 아니라 정상 epoch의 전체 coverage, validation, selected-best final support까지 들어 있다. 이 모든 결과를 GNN 시작의 선행 조건으로 설명한 것은 잘못이다. G4 production catalog 또한 `validate_artifact(payload, 'final')`을 요구하므로 GNN의 final artifact가 필요한 후속 단계가 있다. G3/G4를 통과했다고 바꾸는 것이 아니라, 학습 전 검사와 학습 중·후 판정을 구별한다.

현재 실행 순서:

1. 기존 production cache와 source/config 검증, 짧은 실제 graph/ranking 검사.
2. worker 및 memory batch 측정, 전체 inner-train support 생성, full-support training batch 측정.
3. GNN 전체 40 epochs 및 전체 inner validation, best 모델의 최종 support 생성.
4. GNN 결과/coverage 및 남은 G3 worst-profile·다른 episode 재개 증거 확인, native G4 검사 후 nnU-Net 실행.

**2번의 기존 calibration이 G3의 모든 기준을 대체하지는 않는다.** 다양한 worst profiles, Adam/snapshot 공존 및 다른 next-batch episode 비교 등은 미완료다. G3/G4/G5 전체 합격이나 segmentation 시작을 선언하지 않는다. 아래 명령은 GNN 본학습을 시작하는 명령이며 검사 전용이 아니다.

## 실행 명령

현재 서버는 이미 필요한 `ced3eeb` 코드다. 추가 cache 생성이나 이전 DEBUG checkpoint의 production 재개는 하지 않는다. 현재 conda `nnunet`과 할당된 단일 MIG 환경을 유지한다.

```bash
cd /home/aicompetition06/Medical/HierCP-v22-e1e34bf &&
python -u tools/run_v222_server.py \
  --medical-root /home/aicompetition06/Medical \
  --runtime process \
  --training-objective observed_rank_v1 \
  --feature-coordinates stride4 \
  --cache work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --edge-workspace-mib 256 \
  --output work/v22_gnn40_20260928
```

새 output 경로가 존재하면 기존 결과를 보존하고 실패한다. 캐시는 재사용하고 GNN 가중치는 새로 학습한다. launcher는 graph_DEBUG와 ranking profile_DEBUG를 한 번 수행한 다음 본학습으로 진행한다. profile의 기본 batch32/allocator9GB는 DEBUG 전용이며 production batch는 full-support calibration에서 정한다. 모델·그래프·코호트·40epochs·seed42는 유지한다.

tqdm에 `support_memory` 다음 `epoch 1 optimization`이 나타나면 optimizer 학습 단계다. `support_memory` 중에는 epoch 학습으로 보고하지 않는다. Ctrl+C는 학습 초기화 이후 해당 run에 저장 후 중단을 요청하고 PAUSED를 기다린다. 초기화 전 resources/model/DEBUG 단계에는 이 중단 프로토콜이 없으므로 즉시 전체 종료를 보장하지 않는다. viewer만 닫는 옵션을 주지 않는다. 학습 worker는 기존 launcher가 별도 프로세스로 관리한다.

## 이번 검증

`test_v222_server_runner`, `test_v222_progress`: 허용된 실행 환경에서 14 tests passed. 실제 캐시 재사용 process 계획에 준비 단계가 없고 마지막이 full ranking trainer인지, DEBUG/resume/batch 제한 인수가 본학습에 전달되지 않는지, 실패 시 다음 단계 차단 및 Ctrl+C cooperative pause를 검사했다. orchestration fixture는 의료 학습 증거가 아니다. 첫 sandbox 실행은 오류 표시 후 반환하지 않아 성공으로 집계하지 않는다. 모델/core/runtime 코드는 변경하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 측정 경로를 확인했으며 새 production 측정은 서버 실행 시 수행한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 사용자 제공 MIG/CPU/RAM 및 DEBUG 보고서를 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 변경에 모델 축소 또는 OOM 우회가 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 단위검사의 fixture를 학습 증거로 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 서버 DEBUG 증거 범위이며 전체 학습은 별도다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
