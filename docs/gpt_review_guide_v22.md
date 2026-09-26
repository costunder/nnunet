# v2.2 현재 코드 검토 지도 — observed_rank_v1

이 파일은 현재 검토 경로의 지도다. ZIP manifest의 base_commit이 정확한 소스 기준이며, untracked로 표시된 보조/과거 소스는 GitHub 커밋에 포함되지 않을 수 있다. 현재 작업 중인 root code.txt는 사용자 파일이므로 변경하거나 포함하지 않는다. CT·마스크·그래프 캐시·가중치는 전달하지 않는다.

## 실행 경로와 버전 구분

현재 새 학습은 `tools/run_v222_server.py`의 기본 `process` runtime에서 시작한다. `tools/run_v222_process_runtime.py`가 `tools/run_v222_optimized.py` 실행을 조합하고 runtime/loader/snapshot/review adapter를 설치한다. 새 objective에서는 `hiercp_v222.v1_execution.train`을 `tools/v22_ranking_training.py:train`으로 연결한다. 실제 update/evaluate는 `tools/v22_ranking_steps.py`, 수학적 목적과 필터는 `tools/v22_rank_objective.py`에 있다.

새 기본 설정은 `observed_rank_v1` + `stride4`, overlay는 `config/v22_observed_ranking.json`이다. 원본 `config/prompt_graph_v222_v1_l0.json`과 `config/train.json`은 보존된 base/cache 계약이며 단독으로 최신 학습 목적을 설명하지 않는다. 새 preflight는 `tools/verify_v22_ranking_real_debug.py`를 실행한다.

`run_v222_v1_l0.py` 직접 실행과 명시적 `observation_ce`는 기존 분류 경로다. `run_v222.py` 및 과거 PPR/A*/recipient-only/GraphUNet 제안은 현재 기본이 아니다. 기존 checkpoint의 objective/좌표계 필드가 없으면 CE/legacy로 해석하며 새 rank/stride4로 exact resume하지 않는다. raw graph cache는 재사용하되 새 목표 학습은 새 output에서 시작한다.

## 데이터와 L0 → L1 → L2

고정 split은 `config/split_cp80_fold0.json`: 총131 cases, outer train105/val26, inner train84/val21. 현재 paired cache는 train11,279/val2,823, 총14,102관측이다. 케이스당128비교 위치와 적격 소형 종양 anchor를 사용한다. train 양성527, val135이며 무양성 케이스는 각각19/5다. 소형 기준은 체적 등가 구 직경20mm 이하이며 최대 길이가 아니다. 양성 anchor는 종양 bbox 중심으로 불규칙 종양 내부 voxel과 항상 일치하지 않는다. 비교 위치는 주석상 간이며 과거27.94mm 거리 제한은 없다.

donor는 inner-train에서 query 환자 그룹을 제외해 seed로 배정된다. 기존 캐시는 위치별 donor가 다를 수 있다. 고정 donor의 모든 위치 조합으로 학습된 데이터라고 해석하면 안 된다. case 단위 분할은 고정되어 있으나 실제 동일 환자 재검사 매핑은 확인되지 않았다.

- L0: donor/recipient 각각 단일 CT 채널48³, CNN12/24/32 → 노드 영상 특징32→128 → GATv2 3층/4heads → shell pooling/fusion → paired128D. tumor_surface/source_context/source_liver_surface/target_context/target_liver_surface 노드. tumor interior 노드나 수작업 통계 특징 묶음은 없다. raw CT 중심과 CNN 수용영역의 간 외부 정보까지 제거됐다는 뜻은 아니다.
- 그래프: 물리 좌표와 마스크 기하, k-d/radius 연결, context seed384 + interface +2hop closure의 induced edges.384는 전체 노드 cap이 아니다. 활성 경로에 PPR/A*/뿌리 성장/GraphUNet은 없다.
- L1: data node는 paired128D. 그룹당 관측 두 class label node, 공유 learned seed[2,128], 관계층2개. T/F는 support class 일치/불일치 edge 특성, query U는 정답 미공개 관계다. 기하적 가능/불가능/불명 확률의3-class 분류가 아니다. query 그룹은 support의 recipient/donor 양쪽에서 제외한다.
- L2: 동일 그룹을 가린4head cross-group attention2층, class별 cosine average-linkage clustering, 유효 non-singleton cut 중 silhouette 선택 또는K1, submode별 최소2그룹. detached teacher plan과 live alignment CE를 사용한다. SwAV Sinkhorn의 그대로인 구현이 아니다.

모델 총5,550,806 parameters: L0 4,718,420; L1 relation402,562+seed256; L2 attention132,096+residual297,472. hidden128, 깊이3/2/2, seed42, GNN40epochs, CP80%, accumulation1 유지. physical batch/worker는 실제 loss calibration에 따라 정하며 과거 saved 설정을 임의 축소하지 않는다.

## 최신 변경과 반드시 확인할 한계

아래에 붙인 `docs/v22_observed_ranking_20260927.md`가 새 목적, memory 근사, 평가와 추천 API의 상세 명세다. `tools/v22_rank_recommendation.py:recommend`는 동일 recipient/donor 후보를 모두 scoring한 뒤 GT 마스크로 paste 유효성만 검사한다. 새 API를 nnU-Net 온라인 CP 이벤트에 연결하는 부분은 아직 미완료다. 기존 legacy `v1_local.score_candidates`가 자동 변경됐다고 가정하지 않는다.

`tools/v222_review_contracts.py`는 CNN 좌표의 stride4 수정과 환자 그룹 owner 병합을 제공한다. 원본 입력48격자를 feature12격자 끝점에 단순 대응시키던 오류를 고쳤으며 feature 실제 중심은0,4,…44다. 경계44..47은 기존 border sampling에 따라44로 clamp된다. 이전 수치 동등성 검사는 이 좌표 오류 자체가 옳다는 증거가 아니었다. 정상 worker 소유권, orphan runtime 식별, worker calibration도 함께 수정했다.

## 근거를 읽는 순서

1. `validation/v222_r6/observed_ranking_20260927_DEBUG.json`: 58회귀 검사, 실제 CT physical32 full-model gradient/resume DEBUG, 별도 train8/val2 lifecycle DEBUG, 실제 두 후보 whole-footprint 제외 DEBUG. 전체 학습/추천 효용 검증이 아니다.
2. `docs/v222_independent_review_response_20260927.md` 및 `validation/v222_r6/review_fixes_20260927_DEBUG.json`: 이전 GPT 검토에 대한 수정/미해결 답변.
3. `gpt_handoff.md`, `PATCH_NOTES.md`, `SERVER_V222.md`, `REFERENCES.md`: 최신 항목부터 읽고 과거 설계/실행 기록과 구분한다.
4. `docs/v222_support_snapshot_20260926.md` 등의 속도 결과는 이전 CE/legacy 경로다. 새 rank trainer나 A100 MIG 처리량으로 인용하지 않는다.

전달본 생성은 학습을 실행하지 않는다. 전체40epoch·전체 nnU-Net/CP 비교·A100 MIG 성능은 미검증이다. archive CRC, 원본 SHA256, 분할본 전체 포함 여부, Python 구문 결과는 `BUNDLE_CHECK.json`으로 확인한다. 읽지 않은 과거 코드까지 검토 완료했다고 표현하지 않는다.


## 작업 완료 체크리스트

이 목록은 첨부된 구현 검증 기록 기준이며, 전달 묶음을 만들면서 재학습했다는 의미가 아니다.


- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 시 모델 축소보다 메모리 원인을 우선한다. 이번 실제 검사는 OOM 없이 완료했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사는 실제 CT 검사와 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
