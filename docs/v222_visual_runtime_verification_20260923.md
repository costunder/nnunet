# v2.22 r3 실제 입력 시각화와 VRAM 보정 수정

2026-09-23. 실제 CT 입력과 구현 경로를 확인할 수 있는 조작형 화면을 추가했다. 이번 실행 점검에서 발견한 VRAM 보정 오류 두 가지도 수정했다. 전체 학습의 정확도나 CP 배치 효용을 검증 완료했다고 주장하지 않는다.

## 실제로 잘못된 부분과 수정

1. **GPU 용량을 넘는 보정 실행을 중단하지 않았다.** 16GB RTX5070Ti에서 support 보정 batch128은 peak21.02GB, batch199는32.61GB를 할당한 뒤 부적격으로 기록됐다. 이어진 support 생성은 약0.48graphs/s였다. WDDM 메모리 이동과 일치하는 현상이지만 GPU profiler로 paging 자체를 입증한 것은 아니다. 새 후보 실행 전 peak 예측과 여유 VRAM 검사를 넣고, 부적격 후보에서 보정을 종료한다. Probe 종료 후 gradient와 allocator cache를 정리하고 선택 batch의 처리량을 다시 확인한다. Frozen scorer에도 같은 admission 경로를 적용했다.
2. **첫 수정은 고정 support 비용을 batch마다 중복 계산했다.** Full-support 학습 batch1/2 실제 peak는3.786/4.043GB인데 batch4를14.69GB로 예측해 막았다. 마지막 두 실측의 증가분으로 고정 비용과 query batch 증가 비용을 분리했다. 증가분에는20% 여유를 추가하고, 별도로 실제 가용 VRAM 예산을 적용한다. 감소하는 실측 두 점을 증가 비용0으로 취급하지도 않는다. 이 예측은 모든 shape에서의 절대 보장이 아니며 각 실행 실측과 명시적 OOM 처리를 유지한다.

이 과정에서 실행한 본 작업 Python PID6256과29716만 명령 확인 후 종료했다. 서버·SSH·다른 작업의 프로세스는 종료하지 않았다. 이전 코드와 결과를 보존했고 모델, loss, 표본, 그래프, epoch, CP 확률은 바꾸지 않았다.

## 측정과 현재 실행

- CT 캐시14,102개 완료: inner-train11,279 / inner-val2,823. 전체131case 검사 및 split84/21/26 유지.
- 첫 VRAM 수정 후 전체 support11,279개를16.784graphs/s로 완료했다. inference physical batch16. 이 수치는 optimizer 학습 처리량이 아니다.
- Loader는 동일한11,279개를 workers0/2/4/8로 각각9.856/17.370/21.766/36.452초에 읽었다. CPU 원본 전처리는4worker지만 작은 캐시 patch loader는 실측상0이 빨랐다. GPU를 채우기 위해 근거 없이 worker 수를 높이지 않는다.
- 첫 수정 실행은 full-support batch2에서 optimizer epoch1 step1까지 도달했으나 고정 비용 예측 결함 때문에 중단했다. 완료 학습이나 usable checkpoint로 보고하지 않는다.
- 최종 실행: `work/v222_raw_ct_r3_training_20260923/gnn_vram_affine/`. 전체 GNN40epoch, seed42, gradient accumulation1. Physical/effective batch는 이 실행의 전체 support 보정 후 확정한다. Wrapper PID30208; 실제 자식 PID와 최신 단계는 실행 receipt와 로그로 확인한다.
- 16:39 KST 확인: 실제 Python PID20828에서 support 생성 중이다. 최종 inference 보정은 batch1/8/16이 각각17.829/17.143/17.081graphs/s여서 batch1을 선택했고, 연속 처리량은16.26graphs/s다. 이는 측정에 따른 inference 선택이며 optimizer 학습 batch는 별도다. Batch128 예측23.10GB가14.10GB 예산을 넘어 실행 전에 차단됐다. CT loader0도 같은 전체11279패치의0/2/4/8worker 실측11.258/21.270/24.084/39.480초에 따른 선택이다.
- 로그: `work/v222_raw_ct_r3_20260923/gnn_vram_affine.stdout.log`, `.stderr.log`. 생성한 화면은 명시된 시각의 snapshot이며 자동 실시간 갱신을 주장하지 않는다.
- 전체 GNN40epoch, nnU-Net250epoch, Basic CP80 비교 및 전체 평가 **미완료**. Basic CP80 입력105case는 준비 완료. 두 방법의 donor/크기/crop 차이가 남아 있으므로 whole-method 비교의 범위를 유지한다.

## 기존 입력의 보존과 코드 일치

Original `context/index.json`와 CT 배열은 보존했다. `index_vram_safe.json`는 training/scoring 실행 보정 변경만 연결한 새 provenance다. `index_vram_affine.json`는 그 이후 `batch_admission` 함수만 바뀌었음을 AST로 확인한 새 provenance다. 원본 인덱스 SHA와 이전 source identity, 변경 이유를 각각 기록했다. 학습 시작 시 모든 patch SHA를 다시 확인한다. 가중치에 새 버전 이름을 붙인 것이 아니다.

실행 중 `source_identity()` 대상 라이브러리는 수정하지 않는다. 시각화·문서 도구는 학습 입력과 forward에 포함되지 않는다.

## 사용자가 확인할 수 있는 것

- **실제 CT·L0:** 실제 cache `liver_108:0`과 `liver_108:100`, 48³ CT의 물리 단면 이동. 중심을 가리지 않았음을 확인할 수 있다. 원본 GT overlay는 입력 CT와 별도이며 CNN 채널이 아니다.
- **공간 그래프:** GT와 무관한45,385개 일반 격자 노드 전체를 회전해 확인. 단면에서 노드를 선택하면 실제6mm 반경의3D 이웃과 해당 단면의 연결을 표시. 전체2,383,482 directed edges를 제거하지 않았으며 화면에서는 선택 이웃만 표시한다.
- **L1·L2·CP:** 관측 클래스에 따른 T/F와 query의 U, query case 전체 support 제외, 클래스별 군집 정렬,128개 후보 전체 scoring과 online CP80 경로. 구조도이며 학습된 attention·실제 군집·성능 수치로 꾸미지 않았다.
- **검증·실행:** 동일한 작업의 CPU 처리량, 전체 그래프 GPU DEBUG physical batch 측정, 현재 로그 단계와 미검증 범위.

화면 생성 도구는 `tools/build_v222_visual_data.py`, `tools/render_v222_inspector.py`, `tools/v222_inspector_fragment.html`이다. 첫 도구는 실제 patch/GT SHA를 검사하고 읽기만 한다. `tools/check_v222_inspector.cjs`는 로컬 브라우저에서 단면, GT, 회전, 관계 선택과 폭320px까지의 overflow를 검사한다. 재현 결과는 `work/v222_raw_ct_r3_20260923/visual-check.json`과 PNG에 기록한다.

## 검증 범위

- 기존 단위·회귀89개 통과 기록 유지. 이번 변경 후 새 VRAM6개 + 기존 누수12개 + scorer RNG/backend5개 = **23개 재검사 통과**. 합쳐 서로 다른95개지만95개 전체를 이번에 다시 실행했다고 표현하지 않는다.
- 기존 실제 CT DEBUG에서 CNN/L0/L1/L2 forward→loss→gradient→optimizer 갱신 확인. 이번 runtime 수정은 해당 모델·목표 AST를 변경하지 않았다.
- 시각화는 실제 CT/고정 그래프와 구조도/측정값을 구분한다. 학습한 위치 점수가 CP 성능을 개선하는지는 아직 미검증이다.
- 공개 case 분할이 동일 환자의 재검사 독립성을 증명하지 않으며 원본 주석 누락 가능성도 유지한다.

## 작업 완료 체크리스트

후속 그래프 표시 보완: 사용자가 CT 위 점 표시만으로 그래프를 확인할 수 없다고 지적했다. `tools/export_v222_actual_graph.py`가 실제 `inputs.topology()` 결과를 읽고, 45,385개 노드와2,383,482개 유향 엣지를 무손실 인접 비트맵으로 저장한다. 복원한 연결을 원본과 전수 비교했다. 새 화면의 전체 보기에는119만1741개 양방향 선을 모두 전달하고,1/2/3-hop 보기만 화면상의 부분집합으로 선택한다. 중심 기준 각각57/341/1045개 노드,701/6421/22545개 양방향 연결이다. 브라우저에서 전체 개수·회전·노드 선택·320px 폭을 검사하고 실제 이미지를 확인했다. 학습 코드나 그래프를 변경하지 않았다. 현재 L0는 CT별로 연결을 달리 만드는 구조가 아니라 고정 물리 격자에 CNN 특징을 붙이는 구조다. 특징값·학습 attention은 이 화면에 표시하지 않는다. `actual_graph_verification.json`, `actual-graph-browser-check.json`에 근거를 기록했다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 메모리 및 병목 원인을 먼저 조사하고 모델 규모를 보존했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
# 최신 상태 정정 — 2026-09-23

사용자 지적 후 r3 학습을 중단했다. 아래 실행 기록은 과거 검증이다. 현재 그래프 샘플링 구조의 타당성을 입증한 기록이 아니며 이전 구조도 복구하지 않는다. [설계 감사](v222_graph_design_audit_20260923.md)를 현재 상태의 기준으로 사용한다.
