## 2026-09-24 최신 — 사용자 요청 v1 방식 L0 적용

새 구성은 `config/prompt_graph_v222_v1_l0.json`, 실행기는 `run_v222_v1_l0.py`다. CT-only v1 문맥 그래프 L0(4,718,420 parameters)를 기존 L1/L2/loss/128 후보 점수에 연결하고 실제 CT DEBUG를 완료했다. 전체 학습·native online bank 입력 전환은 아직 미완료다. [구현·검증·제한](../docs/v222_v1_l0_20260924.md)을 따른다. 아래 9천만 CNN과 r4 결과는 이 새 구성의 결과가 아니다.

## 2026-09-24 이전 정정 — 독립 CNN은 승인된 L0 기준선이 아님

90,337,920-parameter CNN은 별도 미학습 진단이다. 실제 L1/L2/CP 학습 경로에 연결된 완성본이 아니며 최신 기준선으로 채택하지 않는다. 뿌리 성장형 구현은 사용자가 지시하지 않았고, assistant의 다중 스케일 CNN 제안도 적용하지 않았다. 이번에는 문서만 정정했으며 모델 수정은 미완료다. `../gpt_handoff.md` 최상단을 따른다.

사용자는 아래 CNN 특징맵 시각화를 반려하고 그래프 요청임을 정정했다. `v22_cnn_l0_20260924`는 실제 실행한 별도 진단으로 보존하며 현재 요청한 L0 그래프의 승인·완료로 취급하지 않는다. 원하는 그래프 구조의 확인 전 임의로 topology를 재정의하지 않는다.

## 2026-09-24 이전 — CNN-only로 해석해 구현한 v2.2 L0

이번 기준선의 고유 ID는 `v22_cnn_l0_20260924`다. `hiercp_v22_cnn/l0.py`, `config/v22_cnn_l0.json`, `tools/run_v22_cnn_l0_one_case.py`를 사용한다. 과거 `hiercp_v22`와 `run_v22.py`는 아래 역사적 그래프 버전이며 이 CNN으로 대체된 파일이 아니다. [구조·실제 한 케이스·미검증 범위](../docs/v22_cnn_l0_20260924.md).

미학습 L0-only 결과는 `work/v22_cnn_l0_one_case_20260924_release/`. 실제 liver_108의 저장된187관찰 위치를 전부 처리했다. 소스/설정과 버전 기록은 `v2.2/cnn_l0_20260924/`에 묶는다. 학습 checkpoint는 생성하지 않았고 과거 가중치와의 호환성을 주장하지 않는다.

## 2026-09-22 이전 v2.22 r2 — 환자 간 prototype 군집 정렬과 References

- 현재 format: `hiercp_v222_cluster_alignment_r2`. r1 소스·문서는 `versions/v2.22/before_cluster_alignment_r2_20260922/`에 보존했다. L0·L1 구조/너비/깊이 및 전체 그래프 규칙은 유지했다.
- L2: query 환자 그룹을 먼저 제외한 뒤 실제 관측 근거가 있는 환자별 L1 label을 클래스별 cosine average-linkage로 군집화한다. 비단독 cut 전체의 양의 silhouette로 K를 선택하고 근거가 부족한 경우 K=1 사유를 기록한다. 환자/그래프를 버리는 cap은 없다.
- 고정 teacher center·assignment는 환자 episode마다 fit하고 batch에서 재사용한다. 기존 L2 2층/128/4 heads에 class-balanced alignment CE를 연결했다. Total loss는 query CE + alignment CE, CP 점수는 live L2 군집 중심의 환자 수 가중 log-sum-exp다. 모든 128개 후보를 점수화하는 online CP 흐름은 유지한다.
- 단독 환자의 미관측 class label은 prototype fit에서 제외한다. Query/validation 정답은 군집 fit에 넣지 않는다. Epoch 로그에 K 후보/선택 사유/점유 수/support 환자 목록/제외 query group/군집 안정성 ARI/중심 유사도·분산을 기록한다.
- 참고문헌을 `REFERENCES.md`에 별도 정리했다. SwAV, DeepCluster, PRODIGY, GATv2, silhouette, 계층 군집, nnU-Net 및 과거 검토 자료에 대해 실제 적용·프로젝트 변형·미사용을 구분한다. **SwAV 그대로의 재현이나 view-only L2라고 주장하지 않는다.**
- 군집 합성 DEBUG 검사 8개 추가, 기존 검사를 포함한 회귀 39개가 모두 통과했다. 정적 검사 18파일과 r1 대비 L0/L1 AST 보존도 확인했다. 실제 CT 3명/6개, 그래프당 24,174 nodes/1,229,900 edges, 파라미터 1,519,063개를 유지한 CUDA batch 2/4에서 gradient·optimizer 갱신을 확인했다. Peak CUDA 0.612/1.100GB. 실제 CT fixture는 support 2명/클래스라 K=1/1이고 다중 군집의 의료 효능 검증은 아니다.
- 전체 GNN 40-epoch·nnU-Net 250-epoch 학습 및 전체 평가 미실행. 실제 임상 군집/소형 종양 성능 개선은 미검증이다. Production identity/annotation manifest, 전체 가림 범위 감사가 여전히 필요하다. 기존 결과를 덮어쓰거나 학습 checkpoint를 만들어 내지 않았다.
- 상세: `docs/pipeline_v222_cluster_r2.md`; References: `REFERENCES.md`; 검증: `versions/v2.22/verification_cluster_r2_20260922/`; 최종 소스 해시와 일치하는 실제 CT 검증: `work/v222_20260922/cluster_r2_final_debug/result.json`. `code.txt`는 현 소스로 갱신하며 export receipt를 같은 검증 폴더에 기록한다. 아래 r1 이하 절은 보존 이력이다.

## 2026-09-22 v2.22 r1 — 관측 관계 학습과 누수 차단 구현

- 사용자 승인에 따라 독립 `hiercp_v222/`, `config/prompt_graph_v222.json`, `run_v222.py`를 구현했다. v2.21은 `versions/v2.21/before_v222_20260922/`에 보존했고 기존 버전 소스/설정은 변경하지 않았다.
- L0: CT 한 채널 CNN(12/24/32) + 공간 GATv2 3층/128차원/4 heads. 수작업 특징·종양 내부/정답 표면 노드를 입력하지 않는다. 학습 분할 전체로 정한 동일한 중심 가림을 보간 전에 적용하고 고정 물리 격자 전체를 사용한다.
- L1: 실제 주석으로 정의한 두 관측 클래스와 T/F support edge를 사용하는 2층 관계 attention. Query의 정답은 loss에만 전달한다. 환자별 자유 latent 16개와 관측 클래스 2개를 혼동하지 않는다.
- L2: query 환자 그룹 전체를 제외한 training support의 환자별 표현을 2층 cross-patient attention으로 정렬한다. Query CE가 L2/L1/query L0에 연결된다. Support L0 memory는 전체 inner-train에서 매 epoch 갱신하는 detached embedding이며 이 학습 선택을 문서화했다.
- 준비/40-epoch GNN 학습/frozen scorer/온라인 CP bank/250-epoch nnU-Net/predict/기존 v5 CSV 평가 CLI 경로를 연결했다. CP 후보 128개 전체 점수와 확률 0.5를 유지한다. 구/DEBUG checkpoint는 production bank에서 거부한다.
- 새 검사 12개 및 회귀 검사 총 31개 통과. 실제 CT 3개/6개 그래프의 결정론 CUDA DEBUG에서 전체 CNN/L0/L1/L2 gradient·optimizer 갱신 및 누수 개입 검사를 통과했다. 별도 native runtime의 V222 트레이너 import도 통과했다.
- DEBUG 그래프당 24,174 nodes/1,229,900 edges; 전체 파라미터 1,519,063개. 전체 그래프를 유지한 attention 계산 분할로 batch 2/4 최대 CUDA 할당량 0.605/1.091GB를 측정했다. Production batch는 full-cohort 자원 보정에서 정한다.
- **전체 데이터 준비·40-epoch GNN·250-epoch nnU-Net·전체 online CP/평가는 실행하지 않았다. 학습 checkpoint나 성능 결과도 만들지 않았다.** 본 준비에는 검증된 환자 그룹/주석 범위 manifest와 전체 가림 범위 감사가 필요하다. 점수는 관측 문맥 순위이며 보정된 종양 발생 확률이 아니다.
- 상세: `docs/pipeline_v222.md`; 검증: `versions/v2.22/verification_20260922/checks.json`; 실제 CT DEBUG: `work/v222_20260922/real_debug4_chunked/`.

# 파이프라인 버전 구분

v2.21은 관계 계약 구현 단계이며 전체 모델은 미완료다. [현 상태](../docs/prodigy_relation_basis_v221.md)를 따른다. v2.2-r5는 `v2.2/before_v221_20260922/`에 보존했다.


## 현재 v2.2 r5 — 폐기된 내부 노드 제거와 정의 정정

- 현재 활성 공간 그래프는 표면/context 계열 5종 node와 13종 edge다. 종양 내부 노드 및 전용 경로는 없다.
- L1 data 단위·T/F/U edge 관계, label16/128차원은 원문 일치/연구 적정성이 검증된 상태가 아니다.
- [현재 정의·변경 근거](../docs/design_contract_corrections_v22_r5.md)
- 변경 전 보존본: `v2.2/before_retired_interior_removal_r5_20260922/`
- DEBUG16개 및 실제 CT 검증: `v2.2/verification_no_interior_20260922/`

## 이하: 이전 버전 보존 기록

## 현재 v2.2 r4 — L2 복구 (2026-09-22)

CT-only CNN L0와 기존 L1을 유지하고, 잘못 삭제했던 기존 L2 정렬/관측 T 전달을 복구했다. L2 forward와 기존 관측 loss 연결을 검증했다. 통계 보조 loss를 제거한 전체 목적함수의 타당성은 별도 미완료이며 전체 학습은 재개하지 않았다.

- [상세 설계/문제 감사](../docs/pipeline_v22_l2_restoration.md)
- 변경 전 r3: `v2.2/before_l2_restore_r4_20260922/`
- 회귀 15개 및 실제 CT CUDA 증거: `v2.2/verification_l2_restore_20260922/`

## 이하: 과거 버전 기록

## 현재 v2.2 (2026-09-22, r3)

**CT-only CNN L0 + 기존 L1 구조 확정**이 현재 버전이다. CNN 없는 통계 특징 r1/r2는 과거 구현으로 보존한다. 가중치 freeze가 아니며 L2 학습 목표와 전체 학습은 아직 미완료다.

- 현재 계약: [v2.2 L0/L1](../docs/pipeline_v22_cnn_l0_l1.md)
- 이전 보존본: `v2.2/before_cnn_only_r3_20260922/`
- 새 검증: `v2.2/verification_cnn_only_20260922/`

## 이하: 과거 버전 설명 보존

사용자의 요청에 따라 **v1 = 기존 파이프라인**, **v2.0 = 폐기된 view-only 구현**, **v2.1 = 기존 환자 간 정렬 구현**, **v2.2 = CNN 없는 원본 관측 특징 GNN**으로 구분한다. 변경 이력의 기준은 [PATCH_NOTES.md](../PATCH_NOTES.md)다.

v2.2는 `run_v22.py` / `hiercp_v22/` / `config/prompt_graph_v22.json`을 사용한다. v2.1의 runtime는 그대로 남겼고, 변경 전 소스·문서는 `v2.1/before_v22_20260920/source.zip`과 manifest에 보존했다. v2.2는 구현·DEBUG 검증 상태이며 전체 학습 완료가 아니다. 상세: [v2.2 계약](../docs/pipeline_v22.md).

아래 표는 기존 v1/v2.1의 비교 기록이다.

| 구분 | v1 (보존) | v2.1 (현재) |
|---|---|---|
| 진입점 | 기존 `run.py` 및 동일 동작 별칭 `run_v1.py` | `run_v2.py` |
| GNN | 기존 `hiercp/` | 새 `hiercp_v2/` |
| L0 | 기존 국소 그래프 인코더 | 기존 KD-tree 국소 그래프와 인코더 재사용 |
| L1 | 기존 patient/region 계층 | 실제 환자별 독립 random trainable label과 data 관계 학습 |
| L2 | 기존 population/prototype 계층 | 환자별 label 공간 정렬 + 관측 T evidence를 U context로 전달 |
| 기존 feedback 실험 | 원래 구현 그대로 | frozen GNN의 정렬 관계 점수 argmax; 별도 실험 |
| 가중치 호환 | 기존 버전 규칙 유지 | v1 가중치를 v2로 변환하거나 재사용하지 않음 |

v1의 내부 `MODEL_ARCHITECTURE_VERSION`은 기존 v5이다. **파이프라인 v1과 내부 모델 revision v5는 서로 다른 번호 체계**다.

`v1/manifest.json`은 변경 전 commit `74dcc2cf03d2d40d1f582223321d96004333f661`의 추적 파일 202개에 대한 SHA-256을 기록한다. `v1/pipeline_v1_source.zip`에는 그 파일들의 사본을 보존했다. 이전 실험 결과 폴더와 미추적 문서는 이동하거나 덮어쓰지 않았다. v2 실행 시 공유하는 v1 코드가 manifest와 일치하는지 검사한다.

2026-09-19 인계 최신화부터 루트 `gpt_handoff.md`/`code.txt`는 현재 문서로 갱신한다. 두 문서의 v1 원본은 변경하지 않은 ZIP에서 SHA를 검증한다. 나머지 v1 파일은 현재 파일까지 원본 SHA를 검사한다. 문서 갱신을 이유로 runtime/config 변경을 허용하지 않는다.

새 구조와 실행 절차는 `docs/pipeline_v2.md`에 기록했다. 이전 GNN PDF는 기존 구조 설명이며 v2 설명으로 간주하면 안 된다.

동일 데이터 두 stochastic view 정렬을 사용한 첫 구현을 **v2.0**으로 기록한다. 사용자 의도와 달라 폐기했다.
그 소스·handoff·code.txt 218개는 `v2/superseded_view_alignment_20260919/source.zip`과 manifest에 보존했다.
현재 **v2.1** format은 `hiercp_prompt_graph_v2_cross_patient_r1`이며 v2.0 artifact를 호환으로 받아들이지 않는다. `run_v2.py`, `hiercp_v2/`, `versions/v2/` 등 기존 경로는 v2 계열 공용 명칭으로 유지한다. 사람용 버전 번호와 저장 format ID를 혼동하지 않는다.

## 작업 완료 체크리스트

2026-09-19 공통 donor/누수 방지 수정 전 17개 파일은 `v2/pre_shared_donor_fix_20260919/source.zip`과 manifest에 보존했다. 현재 증강 계약은 `docs/shared_donor_leakage_v21.md`, 실행·DEBUG 증거는 `docs/local_v21_5070ti_run.md`를 따른다. 이전 비대칭 donor 정책의 cache/bank는 현재 실행에 재사용하지 않는다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사하도록 구현했다. 실제 GPU OOM 실험은 하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] 생산 코드에 dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 GNN 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.


## 작업 완료 체크리스트

2026-09-22 v2.22 구현·DEBUG 검증 범위.

- [x] 서버 또는 원격 세션 종료 위험 명령을 사용하지 않았다. 멈춘 본 작업의 unittest 단일 PID만 확인·보고 후 중단했다.
- [x] 사용자 파일과 기존 실험 결과를 파괴적으로 변경하지 않았다. v2.21 소스와 기존 handoff/patch notes를 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. 승인된 관측 클래스 정의 변경과 공간 노드 의미 변경을 별도로 기록했다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 모든 적격 양성 및 정의된 비교 위치·고정 격자 전체를 처리한다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch와 병렬화 경로를 구현하고 실제 CT DEBUG batch 2/4를 측정했다. 최종 full-cohort batch는 별도 보정한다.
- [x] GPU·CPU·RAM 정보를 확인하고 DEBUG 자원 로그를 남겼다.
- [x] 설정을 낮추는 대신 CUDA 보간의 결정론 호환성을 수정했다.
- [x] DEBUG 설정·자료 범위와 production 실행을 분리했다.
- [x] Dummy 입력을 실제 CT로 보고하거나 가짜 결과/checkpoint를 생성하지 않았다.
- [x] CNN/L0/L1/L2가 forward, loss, gradient, optimizer에 연결됨을 단위 검사했다.
- [x] 실제 실행 설정·새 설계 선택·미검증 범위를 기록했다.
- [x] smoke test와 전체 학습/평가를 구분했다. 전체 학습·평가는 실행하지 않았다.
