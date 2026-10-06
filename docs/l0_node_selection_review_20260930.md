# L0 node selection: actual same-donor graph audit

현재 L0는 영상 특징에 따라 점과 연결을 선택하는 모델이 아니다. CT의 mask/물리 격자에서 후보점을 만들고, 공간 구역별 seed 및 필수 이웃을 선택한 뒤 거리 기반 2-hop closure를 수행한다. CNN 특징은 그 다음 고정된 노드 위치에서 추출된다. 이를 학습된 문맥 탐색 또는 학습된 비격자 topology라고 설명한 것은 부정확했다.

## 실제 한 쌍의 추적 결과

원본: `work/same_donor_learning_DEBUG_20260930/cache/index.json`. 첫 inner_train record `liver_66:1`, donor `liver_1`. 표본을 보기 좋은 결과로 재선정하지 않았다.

| 역할 | canonical context | 초기 선택 | hop1 후 | hop2 후 | 최종 보존율 |
|---|---:|---:|---:|---:|---:|
| donor context | 2,151 | 384 | 1,615 | 2,135 | 99.26% |
| recipient context | 3,315 | 384 | 2,009 | 3,157 | 95.23% |

초기 선택에는 필수 interface 이웃192/251개가 포함된다. 384는 최종 GNN 노드 수 제한이 아니다. context 외 역할까지 포함한 최종 graph는 6,602 nodes, 246,909 directed typed edges다. 13종 관계별 완전히 동일한 endpoint 중복은 모두0개였다. 역방향 관계는 별도의 의미를 가진 typed edge이며, 이를 오류 중복으로 세지 않았다. 계산량은 중복 edge 저장 버그가 아니라 현재 topology 자체에서 나온다. 이 수치만으로 각 연결이 의미적으로 불필요하다고 판정하지 않는다.

`tools/trace_l0_node_growth_debug.py`는 실제 `_select_context`의 반환을 바꾸지 않고 각 단계의 ID를 추적한다. 추적 없는 original materialization과 각 역할의 grid 및 모든 edge_index가 정확히 동일함을 검사했다. 학습 실행이나 모델/그래프 규칙 변경은 없다. CPU geometry 분석이며 CNN/GNN 속도 benchmark가 아니다.

결과: `work/l0_node_growth_same_donor_DEBUG_20260930/report.json`, 그림 `node-growth.png`. 그림은 donor/recipient **context 노드만** 표시한다. 파랑은 초기 선택, 주황은 hop1 추가, 분홍은 hop2 추가다. attention/학습 중요도 heatmap이 아니다. 각 행의 세 그림은 동일 좌표와 카메라를 쓴다.

## 바로잡을 설계 경계

1. CNN이 읽는 영상 표본과 GNN이 message passing하는 대표 노드를 구분해야 한다. 인근 표본을 읽었다는 이유로 전부 다음 graph의 노드로 추가하면 현재 문제가 반복된다.
2. 노드 선택에 영상 특징이 들어가는지, 선택/집계 모듈이 실제 ranking loss의 gradient를 받는지 별도로 검증해야 한다. 단순 균일/FPS 샘플링에 CNN 특징을 붙이는 것만으로 학습형 선택이라고 하지 않는다.
3. 역할별 국소 이웃 특징을 집계해 대표점을 갱신하는 경로는 검토할 수 있다. 지점 수만 자르거나 기존 hop 값을 몰래 줄이는 구현은 하지 않는다. 문맥 범위·누락률·연결성·계산량을 함께 측정해야 한다.
4. current CNN/SAGE width/depth, L1/L2, 전체 observations, donor 비교, 후보128, 원본 paste-mask 검사는 별개다. 이번 구조 점검을 이유로 동시에 바꾸지 않는다.
5. 비격자 모양 자체는 성능 증거가 아니다. 선택된 위치의 움직임/특징 반응, gradient, 실제 update 시간, 동일 heldout ranking을 구분한다. 무작위 가중치로 만든 모양을 학습된 중요 영역이라고 표시하지 않는다.

## References and applicability

- PointNet++ §3: https://arxiv.org/html/1706.02413#S3 . FPS centroids와 이웃 집계를 계층적으로 사용한다. FPS 자체는 학습된 영상 중요도 선택이 아니다. CT의 정답 설계로 승격하지 않는다.
- Superpoint Graph §3: https://arxiv.org/html/1711.09869#S3 . 점을 영역으로 묶고 영역 단위 관계를 처리한다. point-cloud geometric partition의 CT 내부 문맥 적합성은 별도 문제다.
- PointASNL §3.1: https://arxiv.org/html/2003.00492#S3.SS1 . 초기 FPS 표본 주변의 feature attention과 학습된 가중 집계로 대표 좌표와 특징을 조정한다. 고정 좌표 선택과 feature-aware adaptive sampling의 구분에 직접 관련된 참고다. 본 프로젝트에서 이 논문 전체를 구현하거나 CT/CP 성능을 검증한 것은 아니다. kNN 등 논문의 구성요소를 기존 EZ-SP 함수의 대체물로 넣지 않았다.

이 작업의 완료 범위는 실제 topology 원인 추적·시각화·수정 경계 명시다. **새 L0 구현 완료 또는 현행 L0 설계 검증 완료가 아니다.** 이전 same-donor 학습 수정은 순위 비교 조건의 수정이며 이 node-selection 문제까지 해결한 것으로 읽어서는 안 된다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험 명령을 사용하지 않았다.
- [x] 기존 결과를 파괴적으로 변경하지 않았다. 별도 결과 디렉터리를 사용했다.
- [x] 모델 깊이와 너비를 축소하지 않았다.
- [x] 실제 추적 graph의 노드와 edge를 변경하지 않았다.
- [x] 숨겨진 cap/subset을 추가하지 않았다. 첫1pair 분석은 명시적 DEBUG다.
- [x] 모델 학습 대신 CPU geometry 추적만 실행했다. 기존 loader4workers를 사용했다.
- [x] 입력 RAM cache4GiB 및 RSS24GiB 예산을 설정했다. GPU 학습은 미실행이다.
- [x] OOM 회피를 위한 모델 축소를 하지 않았다.
- [x] DEBUG 결과를 production 학습 결과와 구분했다.
- [x] dummy 출력 또는 fallback을 사용하지 않았다.
- [ ] 새로운 선택 모듈의 forward/loss/gradient/optimizer 연결: 아직 새 모델을 구현하지 않았다.
- [x] 실제 record, 표시 범위, 계측 방법과 제한을 보고했다.
- [x] topology 일치 검사와 전체 학습/성능 검증을 구분했다.
