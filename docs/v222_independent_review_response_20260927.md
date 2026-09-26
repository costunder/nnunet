# v2.22 r6 독립 검토 대응 — 2026-09-27

첨부 검토의 실행 guard·CNN 좌표·다중 검사 환자 grouping 반례는 실제 코드에서도 재현됐다. 이전 수치 동일성 검사는 기존 잘못된 좌표 해석까지 보존했으므로, 그 검사만으로 입력 위치의 정확성을 승인했던 판단은 잘못이다. 이번에는 좌표 자체의 정답을 가진 독립 반례를 추가했다.

대상은 `tools/run_v222_server.py --runtime process` → `tools/run_v222_process_runtime.py` → paired v1 L0 + v2.22 L1/L2다. 본 수정은 연구 성능 승인이나 전체 학습 완료가 아니다. 서버의 실행 중인 프로세스·가중치·원본 캐시는 변경하지 않았다.

## 지적별 조치

| 지적 | 확인 및 처리 | 검증 범위 |
|---|---|---|
| P1 CNN 입력 위치와 특징 격자 대응 | 수정. 실제 CNN의 kernel/stride/padding에서 특징 중심과 간격을 계산하고 입력 좌표→특징 좌표를 명시적으로 변환 | 독립 좌표장, 실제 CT L0, 전체 모델 backward |
| P1 정상 server worker가 자기 자식 trainer를 차단 | 수정. 기록된 PID/생성 시각에 더해 실제 자기 조상이며 정확히 같은 output의 `--worker`인 경우만 허용 | 실제 parent/child subprocess 반례, 다른 output/조상 아닌 worker 거부 |
| P2 process runtime 고아 trainer 누락 | 수정. 세 trainer 진입점 모두 exact-output 검사에 포함 | 각 진입점의 중복 차단 및 무관한 프로세스 비개입 검사 |
| P2 다중 검사 case owner를 환자로 취급 | 수정. support 소비 시 같은 `patient_group`을 한 owner로 합침. query의 recipient 및 donor 양쪽 group 배제 유지 | A1/A2/B1 반례, 동일 환자 병합, 기존 case별 경로 bitwise 보존 |
| P3 worker 튜닝이 `.make()`를 측정 | 수정. production `.batches()`의 producer/prefetch/pinning 경로로 cold/warm 측정 | 실제 CT 그래프 16개×4 batch, workers 0/2/4/8 |
| donor 적합성·CP 효용 | 아직 입증되지 않음. 현재 loss를 임의 변경하지 않음 | 아래 별도 연구 검증 필요 |
| paired online CP→nnU-Net | 미완료 상태 유지 | GNN smoke를 end-to-end 완료로 표현하지 않음 |

## 좌표 수정의 의미와 checkpoint

48³ 입력에 대한 현재 CNN의 두 stride-2 convolution은 12³ 특징 격자를 만들고, 명목상 중심은 각 축 `0,4,…,44`다. 기존 코드는 입력 정규화 좌표를 특징 격자에 그대로 적용해서 입력 중심23.5에서22.0 위치에 해당하는 특징을 읽었다.

수정은 `입력 정규화 좌표 → 입력 voxel → (voxel-origin)/stride → 특징 정규화 좌표` 순서다. 독립적인 선형 좌표장을 사용한 검사에서23.5를 요청하면23.5를 읽는다. 특징 중심의 마지막 위치44를 넘어선 입력44~47은 기존 border 규칙으로 마지막 특징을 읽는다. CNN의 명목상 중심은 receptive field 전체가 한 점이라는 의미가 아니다.

`tools/v222_review_contracts.py`가 실행 시 모델에 변환을 연결한다. frozen 연구·그래프 생성 소스와 설정은 그대로이므로 기존 raw CT graph cache의 provenance를 유지한다. 새로 생성한 corrected encoder의 sampler는 설치 context 종료 후에도 인스턴스에 남는다.

- 새 process 학습: `feature_coordinates=stride4`. 모델 초기화부터 수정된 특징으로 학습하고 support를 새로 계산한다.
- 기존 checkpoint 재개: 필드가 없으면 `legacy`. 가중치/Adam/RNG/지원 임베딩/학습 위치와 기존 입력 해석을 유지한다. 알려진 배포 소스 해시만 migration 대상으로 허용한다.
- `legacy` checkpoint에 `--feature-coordinates stride4`를 붙인 exact resume는 오류로 거부한다. 반대 전환도 거부한다. 좌표 수정은 단순한 속도 최적화가 아니다.
- 같은 환자 여러 case로 구성된 legacy support/cluster plan은 새 group 해석으로 재개하지 않고 별도 학습을 요구한다. 현재 공개 데이터의 case별 고유 group 경로는 동일하다.
- rolling checkpoint의 `execution_policy` 및 epoch/final 모델 artifact의 `feature_coordinates`, `support_task_contract`에 의미를 기록한다. 초기 `.workspace.json`은 원래 base wrapper가 작성하며 feature 필드가 없을 수 있으므로 embedded checkpoint와 initialization/execution 보고서가 의미 판정의 기준이다.
- fresh process run의 graph DEBUG/profile DEBUG에도 동일한 좌표 flag를 전달한다. DEBUG profile의 loader는 기존 optimized backend이며 process throughput benchmark가 아니다.

직접 `run_v222_v1_l0.py` 또는 `--runtime optimized/legacy`로 실행하는 보존 경로에는 이 corrected adapter가 적용되지 않는다. 수정된 학습의 실행 명령은 `SERVER_V222.md` 최상단을 따른다. 기존 root 소스를 직접 읽고 corrected 동작이라고 추정하면 안 된다.

## 실제 확인한 범위

RTX5070Ti 16GB, CPU 8 physical/16 logical, RAM64GB 환경을 확인했다. DEBUG와 최종 설정을 분리했다. 48개 관련 회귀 검사를 통과했다. 실제 CT 학습 검사는 전체5,550,806 parameters, physical32, accumulation1, bf16,256MiB edge workspace, PyTorch allocator9GB 조건이었다.

- query32개의 실제 그래프:346,135 nodes /23,857,190 edges. 그래프별 node/edge 제한이나 단순화 없이 사용했다.
- support는 실제 train3개 group의 T/F6개 관측만 사용한 명시적 DEBUG다. query group은 support recipient와 donor 양쪽에 없음을 확인했다. 전체11,279 support를 사용한 연구 성능 검사로 보지 않는다.
- corrected/legacy의 같은 가중치 L0 결과는 달랐다(max absolute difference0.388671875). 좌표 수정의 실제 forward 연결을 확인한 것이며 개선된 정확도의 증거는 아니다.
- forward/loss/backward/Adam update가 완료됐고 모든 학습 파라미터의 gradient가 존재하며 유한했다. peak allocated6,295,574,016 bytes, reserved6,582,960,128 bytes. 소규모 support의 단일 update 용량 검사이며 전체 학습 VRAM 상한을 보장하지 않는다.
- 별도 실제 CT corrected 연속 실행과 corrected checkpoint 재개 간 loss/전체 model/Adam 및 partial support memory가 bitwise 일치했다. legacy와 corrected가 동일하다는 의미는 아니다.
- production loader 경로의 실제 calibration에서 4batch warm시간은 workers0=1.416s,2=0.870s,4=0.762s,8=0.768s였다. 이 로컬 probe에서는4가 선택됐으나4와8 차이는 작고, GPU overlap이나 A100 서버 최적 worker 수를 입증하지 않는다. 기존 resume의 saved workers8은 변경하지 않는다.
- worker 후보별 producer 수는1/2/4/4, producer당 decode는0/1/1/2다. 후보 간 pool을 정리해20% RAM cache 예산이 후보 수만큼 중첩되지 않도록 했다. 선택된 production pool은 단계 사이 유지한다.

정확한 실행 시점의 소스 해시와 결과는 `validation/v222_r6/review_fixes_20260927_DEBUG.json`에 기록한다. 변경 전후 전체 support 속도 비교는 이전 legacy 좌표의 DEBUG 결과이며 이번 corrected 정확성 검사의 성능 결과로 전용하지 않는다.

## 아직 해결됐다고 말할 수 없는 연구 문제

현재 label은 recipient의 기존 종양 관측에 의존하고 donor는 label-blind하게 선택된다. 중심 CT를 가리지 않는 것은 사용자 승인 사항이었다. 따라서 분류가 잘되더라도 donor와의 적합성보다 recipient의 기존 종양 존재를 이용할 가능성이 있다. 실제 학습 모델이 donor를 무시한다고 확인한 것은 아니다.

필요한 검증은 ①동일 recipient와 동일 유효 후보에서 donor만 바꾸는 순위/점수 변화, ②recipient-only 및 중심 CT 노출 여부를 통제한 비교, ③no-CP/random-CP/제안 CP를 동일 split·seed·CP80% 등 합의된 조건으로 비교하는 nnU-Net segmentation 실험이다. 이번에는 이 실험을 실행하지 않았으며 loss·label·CT 중심 노출·모델 깊이·graph 규칙을 바꾸지 않았다.

paired online adapter가 아직 없으므로 현재 GNN 점수가 nnU-Net CP 이벤트에서 실제 사용된다고 말할 수 없다. corrected checkpoint를 사용할 후속 inference adapter도 저장된 feature contract에 맞게 encoder를 생성해야 한다. 높은 관측 분류 AUROC를 CP 효용이나 소형 종양 segmentation 개선으로 보고하면 안 된다.

현재131-case에서 환자 재검사 중복이 실제 존재하는지는 미확인이다. grouping 반례 수정은 환자 독립성 확인을 대신하지 않는다. 알려진 source run의 guard는 전역 GPU lock이 아니며 무관한 GPU 프로세스 점유나 동시에 시작하는 모든 race를 막지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 검사는 OOM 없이 완료했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 독립 좌표 반례/모의 lifecycle 검사는 단위 검사로 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 CT DEBUG 범위에서 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 전체 학습·평가·서버 적용은 하지 않았다.
