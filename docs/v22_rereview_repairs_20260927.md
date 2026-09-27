# v2.2 재검토 R01–R06 수정 — 2026-09-27

검토 기준 `cb3154c5f8c3494c92ae929c60a4bb6d993f1719`. 첨부 텍스트 전문을 읽고 남은 조건부 admission 결함을 수정했다. 텍스트에 링크된 별도 sandbox 보고서·재현 ZIP은 실제 첨부되지 않았으므로 읽거나 재실행했다고 주장하지 않는다. 정상 trainer가 손상된 checkpoint를 만들었다는 증거와 검증 누락을 구별한다.

## 수정 매핑

| 지적 | 실제 변경 | 확인 |
|---|---|---|
| R01 Adam | 이름·순서·shape·dtype·hyperparameter 계약과 CPU snapshot optimizer hash. step0 빈 state와 update 후 모든 parameter moment를 구별. NaN/Inf·shape·step·group 불일치 거부 | 실제 CPU Adam 다음 update 일치, 손상 반례, 실제 GPU process 연속/재개 비교 |
| R02 best/cursor | state.best=embedded.metric=meta.metric, best epoch≤검증 완료 epoch. 데이터와 동일한 batch 순서로 seen·cursor·step·loss history 확인. phase별 support prefix·plan 유무·terminal 전이 검증 | 미래 best·terminal validation·cursor 모순 거부. 마지막 epoch 직후 optimization 전이는 허용 |
| R03 calibration | objective/features/support/geometry/source/cache/model/cohort/device/runtime 검증. nonempty 목록·유한한 측정·accepted/executed·선택 batch/worker의 실제 측정값 확인 | 정상 fresh/reuse 실제 process 성공. 잘못된 보고서는 재사용 차단 |
| R04 placement | 모든 후보를 실제 recipient spacing/affine과 대조. center 외 donor·transform·CT/mask hash·anchor·frame을 동일 event로 결속 | 둘째 후보 변조 및 순서 반전, 대칭 mask의 transform 변조 거부 |
| R05 graph 내용 | recipient CT hash, 간 union hash, 실제 graph payload hash와 materialization 계약을 factory에서 저장하고 API에서 확인 | CT/간 union/target patch 변조 거부. 간 union을 유지하는 2→1 종양 label 변경은 허용 |
| R06 workers0 | 0은 CPU materialization만 serial, GPU graph batching은 유지. 양수는 명시적 thread pool | 실제 checkpoint→recommend에서 0/1/2/4/8 결과 완전 동일 |
| 빈 요청 | recipient/grid/group/coverage/batch/worker 검사를 빈 후보 반환보다 먼저 수행 | 정상 empty는 no-op, invalid group/None recipient/NaN coverage는 오류 |
| Windows 검사 | Windows redirector 검사를 Windows에서만 실행. 관련 CPU 검사의 임시 폴더를 work/ 존재에 의존하지 않게 변경 | Windows 로컬 회귀. Linux 실행은 이번 환경에서 미실행 |

이번 구현의 CP event는 **동일 donor·동일 transform·여러 center** 계약이다. 현재 API는 후보별 transform 탐색을 구현한 적이 없으며, 공통 footprint를 필터에 전달하던 기존 의도를 명시적으로 검사한다. transform 검색이라는 새 연구 설계는 추가하지 않았다.

내용 hash는 accidental mismatch/stale graph를 검출하는 일관성 검사다. 악의적으로 payload와 hash를 모두 재작성하는 공격에 대한 서명/인증 체계라고 설명하지 않는다. 전체 tumor-label hash를 forward binding에 넣지 않았다. 간 union 기하에는 여전히 GT label1/2가 사용된다.

## 실제 검증

출력은 `work/v22_rereview_20260927_DEBUG/`에 새로 저장했다. 기존 결과를 덮어쓰거나 삭제하지 않았다.

- RTX5070Ti 16GB, 8 physical/16 logical CPU, 시작 시 가용 RAM42.52GiB 확인. 새 raw-content binding을 포함한 실제 CT graph10개를 재생성했다.
- full model5,550,806 parameters, train8/val2, physical2, support8, DEBUG1epoch/4updates. production40epochs·전체 cohort·모델 깊이·그래프 규칙·seed42·CP80%·donor 배정·loss는 변경하지 않았다.
- fresh process calibration → 첫 update durable pause → 실제 Adam/cursor admission → 나머지 updates → validation → best/final support 완료.
- 같은 calibration을 재사용한 중단 없는 run과 중단 후 재개의 **model, Adam, Torch/NumPy/Python/CUDA RNG, final support 모두 bitwise 일치**. DEBUG 생성기의 비Torch RNG도 동일하게 맞춘 대조이며 전체 학습 검증은 아니다.
- 실제 CT의 기존 후보2개와 추가 DEBUG 유효 paste 대조 위치1개로 추천했다. 추가 위치는 production proposal 규칙이 아니라 성공적인 paste 검사용 fixture다. raw 1위는 간 coverage, raw2위는 종양35voxel 겹침으로 제외되고 raw3위가 선택됐다. raw score를 -inf로 바꾸지 않았다.
- 선택 위치에 **2,368 voxel 실제 raw CT/label paste**를 수행해 좌표·CT 값·외부 voxel 불변을 검사했다. 이전 같은 크기 mask의 no-op 검사와 구별한다. nnU-Net resampling/최종 입력 검증은 아니다.
- workers0/1/2/4/8의 추천·순위·eligibility가 완전 동일했고, 고정 graph/CT/간 union에서 종양 label만 2→1로 바꾸면 score/rank는 동일하고 eligibility만 변했다.
- 관련 단위·회귀 **72개 통과**. 실제 GPU 저장 checkpoint의 정상 CPU admission을 확인한 뒤 Adam/best/cursor/partial support 변조12종을 거부했다. 손상된 optimizer로 GPU update를 실행하지 않았다.
- calibration CE/legacy/runtime/malformed dict/empty list/미측정 batch/NaN7종을 **실제 process**에서 첫 checkpoint 이전에 거부했다. unknown runtime migration을 승인한 경로는 없다.
- 마지막 epoch final-memory의 실제 portable resume도 통과했다. rolling 파일 하나만 복사하고 외부 best 파일 읽기를 금지한 상태에서 selected weights와 final support가 bitwise 일치했다.

최종 source/runtime SHA 및 실제 출력 hash는 `validation/v222_r6/rereview_repairs_20260927_DEBUG.json`에 기록한다. 독립 검토자가 실행했다고 적은 수치와 이번 직접 실행 결과를 합산하지 않는다.

## 이전 상태와 남은 범위

artifact는 `observed_rank_artifact_v3`로 변경했다. 이전 artifact v2에 새 hash/contract를 임의로 붙여 exact resume하지 않는다. 원본 CT/geometry 규칙 자체는 바뀌지 않았지만 graph에 새 content binding이 필요하며 source provenance도 바뀌므로 기존 cache를 그대로 새 것으로 표시하지 않는다. 이번 DEBUG는 실제 factory로 재생성했다.

Calibration은 **현재 runtime과 동일한 측정만 재사용**한다. 승인된 execution-only migration 목록이 없으며 unknown runtime을 거부하고 재측정한다. optimizer hash는 이미 CPU로 복사된 snapshot에서 계산하고 같은 update 번호의 no-update support pass에서는 재사용한다. epoch 가중치 저장/gradient 의미를 변경하지 않는다.

pair 비균등 가중, CE/rank gradient 관계, 종양 본체 shortcut, fixed-donor 효용, 전체 support worst-batch calibration은 별도 연구/규모 검증 항목으로 남는다. 원래 estimator의 중복 제거·loss/2·L1/L2 detach를 적용하지 않았다.

G3 새 전체 코호트·full support 실행, G4 native bank/RPC/adapter/trainer 및 nnU-Net 최종 입력 연결, G5 CP 효용 비교는 미완료다. raw paste 성공을 native online CP 완성으로 표현하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. DEBUG 표본은 분리했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. DEBUG 측정이며 전체 규모 승인 아님.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한다. 이번 DEBUG OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 production에 사용하지 않았다. CPU fixture 검사는 별도 표시했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 full-model DEBUG update 확인.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
