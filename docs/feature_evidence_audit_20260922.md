# 수작업 특징·L2 학습 목표 근거 대조

2026-09-22 KST. 대상: 로컬 v2.1 및 작업 중단된 v2.2-r2. 이번 작업은 원문·논문·코드 대조와 기록 정정이다. 모델 수정, 학습 재개, 새 특징 채택은 하지 않았다.

## 결론

**수작업 특징이라는 방법 자체에는 선행 연구와 비교 실험 근거가 있다. 그러나 현재 16차원/24차원 조합과 이를 이용한 L2 정렬 목표가 종양 Copy-Paste에 유효하다는 직접 근거는 이번에 확인한 자료에 없다.**

따라서 “전부 근거 없이 발명한 값”, “논문에서 검증된 현재 구성”, “모두 삭제해야 한다”는 세 주장 모두 이 조사 결과로 정당화할 수 없다. 계산 정의, 다른 과제에서의 효과, 현재 CP 과제의 효과를 구분해야 한다. 여기서 근거 미확인은 관련 문헌 전체에 근거가 존재하지 않는다는 증명이 아니다.

## 사용자 제공 원문과 실제 합의의 범위

| 자료 | 확인 위치 | 실제 내용 | 뒷받침하지 않는 주장 |
|---|---|---|---|
| [CNN 없는 그래프에 관한 첨부 원문](C:/Users/user/.codex/attachments/93fae564-a423-4001-9642-5c5ac0f94e4b/pasted-text.txt) | 94–122행, 185–199행 | 상대 좌표·CT 강도·평균/표준편차/분위수·국소 강도 변화·간 표면 거리·문맥 구분을 **제안 설계**로 예시. 특정 논문 구성을 그대로 옮긴 것이 아니라고 명시. CNN 유무 등의 비교 실험 제안 | 현재 16/24차원이 논문에서 검증됐다는 주장, 그 정확한 창 크기·정규화의 타당성, 사용자 후속 이의 제기를 무시한 채 확정 요구사항으로 취급하는 것 |
| [L2에 관한 첨부 원문](C:/Users/user/.codex/attachments/c329cbfe-af55-4be8-8803-8d7dfc7e0bbc/pasted-text.txt) | §2·§3·§6·§8 | 서로 다른 실제 task의 국소 latent label 공간 정렬 및 희소 positive evidence 전달. 미관측 위치를 자동 negative로 취급하지 않음. 원본 GT를 추정만으로 바꾸지 않음 | CT 통계의 cosine similarity를 정렬의 teacher로 삼으라는 지정. 정확한 손실식·특징 조합·계수의 실증적 타당성 |
| 이번 대화에서 사용자의 정정 | “기하/통계 특징이 네가 임의로 만든거지 내가 만들라고 한적이 있음?” 등 | 임의 특징 선택과 근거 없는 대체 설계에 명시적으로 이의 제기 | 이전 붙여넣기 글을 이유로 새 특징을 임의 추가해도 된다는 해석 |

첨부 글은 논문 원문이 아니라 제안·설명문이다. 그 안의 제안과 인용 논문이 실제로 검증한 범위를 따로 확인했다. “첨부에 통계 특징 예시가 있다”는 사실을 사용자 탓이나 현재 구현의 포괄적 승인 근거로 사용하지 않는다.

## 논문·표준에서 확인한 범위

| 근거 | 본문 위치와 확인 결과 | 현재 프로젝트에 대한 한계 |
|---|---|---|
| [Superpoint Graph, Landrieu & Simonovsky](https://arxiv.org/html/1711.09869) | §3.1, §3.3, Appendix A/C. 선형성·평면성·산란성·수직성·고도를 사용. S3DIS 6-fold의 Table 6(b): 기하 특징 미사용 58.4, 사용 62.1 mIoU, 차이 +3.7 %p | 수작업 특징과 학습 모델의 결합이 실제로 효과를 보인 **다른 과제의 근거**. 우리의 CT 평균/표준편차/곡률 묶음이나 CP 효과의 증거는 아님. Table 6(b)는 입력 기하 특징 비교이며 별도의 superedge 특징 제거 실험과 혼동하지 않음 |
| [PointNet++](https://arxiv.org/html/1706.02413) | §3.2. 좌표와 점 특징 입력, 상대 좌표, 반경 이웃, local PointNet으로 국소 표현 학습 | 공간 이웃과 학습 표현의 선행 사례. CT 16/24차원 사전 계산 특징을 요구하지 않음. 현재 모델은 PointNet++ 재현이 아니며 논문의 sampling/이웃 제한을 이번에 이식하지 않음 |
| [DGCNN](https://arxiv.org/html/1801.07829) | §3, §4.5. EdgeConv로 이웃 관계 특징 학습. S3DIS 입력은 XYZ·RGB·정규화 좌표의 9차원. 국소 곡률/법선 등을 쓰는 별도 비교 baseline도 기술 | 현재 CT 통계 묶음의 직접 근거가 아님. 모델 이름을 인용했다고 현재 MLP+GAT가 DGCNN 재현이 되는 것도 아님 |
| [IBSI intensity statistics](https://ibsi.readthedocs.io/en/latest/03_Image_features.html#intensity-based-statistical-features) | Mean intensity 및 Intensity variance 항목. 모집단 평균·분산의 정의. 표준편차는 그 분산의 제곱근 | **계산 정의의 근거**. 노드별 마스크된 4/12/28 mm 반폭 상자·부분 복셀 가중치·CT clipping·L2 teacher의 임상/CP 유효성을 인증하지 않음 |
| [Voxel2Mesh](https://arxiv.org/html/1912.03681) | §3. CNN의 volumetric feature를 mesh 정점에 전달하고 graph 연산으로 처리 | CNN→그래프 특징 전달의 선행 사례. 현재 수작업 특징 묶음이나 특징 기반 L2 teacher의 근거가 아님 |
| [PRODIGY](https://arxiv.org/html/2305.12600) | §2.3/§3.1 및 Appendix C. Task의 class label node, support의 정답 관계 T/F, query에 대한 label→data 메시지, edge 속성을 반영한 attention. Appendix A에는 masked node attribute reconstruction이 별도 존재 | 랜덤 label 초기화와 data–label attention의 선행 근거는 있음. 환자별 자유 latent slot 사이의 CT 통계 cosine teacher는 정의하지 않음. attribute reconstruction의 존재만으로 현재 116/144차원 복원 및 L2 목표가 논문 재현이라고 할 수 없음 |

위 SPG 수치는 S3DIS의 mIoU다. 종양 Dice, 소형암 recall 또는 이 저장소 실험 결과에 옮겨 적으면 안 된다.

## v2.1: 실제 16차원 입력 전수 대조

생성: [hiercp/local.py](../hiercp/local.py) `_node_features` 168–216행. 계산: [hiercp/spatial.py](../hiercp/spatial.py) 383–404행 및 516–537행. 입력 연결: [hiercp/model.py](../hiercp/model.py) `forward_graph` 733행 이후에서 16차원을 CNN 표본 특징 32차원과 결합한다.

아래 표의 판정은 **현재 CP 과제에 대한 직접 효과는 모든 항목에서 미검증**이라는 전제다. “정의 가능”은 “필수”를 뜻하지 않는다. 인덱스는 0부터 센다.

| 인덱스 | 특징 | 폭 | 출처·성격 | 현재 구현의 제한 |
|---|---|---:|---|---|
| 0 | 정규화 CT 강도 | 1 | 관측값; 첨부 제안에 존재 | clipping/정규화 및 노드 위치 선택까지 원본 그 자체는 아님 |
| 1 | 주변 CT 평균 | 1 | 첨부 제안; 평균의 수학적 정의는 IBSI와 관련 | `uniform_filter(size=3)`의 복셀 창. 환자별 같은 mm 범위 보장 없음 |
| 2 | 주변 CT 표준편차 | 1 | 첨부 제안; 모집단 분산의 제곱근 | 동일 3복셀 창. 경계 확장 `nearest`; 현재 창/관측 영역의 타당성 미검증 |
| 3 | CT 기울기 크기 | 1 | 첨부의 국소 강도 변화라는 넓은 제안 | `np.gradient(ct_norm)`에 spacing 없음. 복셀당 변화이며 mm당 변화로 해석하면 오류. 0–2 clipping도 프로젝트 선택 |
| 4–6 | 상대 좌표 | 3 | 첨부 및 PointNet++/DGCNN의 일반 선행 사례 | 현재 기준점은 native ROI 중심. 정규화 반경과 다른 특징에 대한 상대 가중은 프로젝트 선택 |
| 7 | 종양 signed distance | 1 | 마스크로 계산한 기하량 | 반경으로 나눈 뒤 −2–2 clipping. 이 값을 후보 적합성 입력으로 써야 한다는 직접 근거 미확인 |
| 8 | 간 내부 깊이 | 1 | 첨부의 간 표면 거리와 관련 | 물리 EDT를 80으로 나누고 0–2 clipping. 이 스케일의 근거 미확인 |
| 9–11 | SDF 법선 | 3 | 수학적 형상 표현 | 여기의 법선 계산은 spacing 사용. CT 기울기의 spacing 문제와 혼동 금지. CP 유효성은 별개 |
| 12 | 곡률 근사 | 1 | 정규화 SDF gradient의 divergence | spacing 사용, 3으로 나누고 clipping. SPG의 PCA 기하 특징과 같은 특징이 아님 |
| 13 | shell 구분 값 | 1 | 첨부의 near/far 문맥 구분과 관련 | 연속 거리 대신 구간 번호를 입력하는 방식과 경계는 프로젝트 선택 |
| 14 | surface flag | 1 | 노드 역할 메타데이터 | 다른 node type 표현과의 중복/필요성 미검증 |
| 15 | branch flag | 1 | source/target 역할 메타데이터 | 생물학적 특징 아님; 필요성 미검증 |

총 16차원이다. 공간 그래프 구성에 좌표를 사용하는 사실만으로, 같은 좌표를 노드 입력과 복원 목표에도 넣어야 한다는 결론은 나오지 않는다.

## v2.2-r2: 현재 24차원과 문서 불일치

설정은 [config/prompt_graph_v22.json](../config/prompt_graph_v22.json)의 `release=v2.2-r2`, `physical_observation_boxes_mm_masked_r2`다. [hiercp_v22/features.py](../hiercp_v22/features.py) 13–19행에 목록, `dictionary()`에 출처 한계가 있다. [hiercp_v22/local.py](../hiercp_v22/local.py) 86행 이후는 `node.observed`를 입력으로 쓴다.

| 인덱스 | 특징 | 폭 | 현재 근거 판정 |
|---|---|---:|---|
| 0 | CT 강도 | 1 | 관측값 기반; 현재 입력 설계 효과 미검증 |
| 1–9 | 4/12/28 mm 반폭 상자 각각의 평균·표준편차·관측 비율 | 9 | 평균/분산은 정의 근거 있음. **반폭·상자 형태·마스크·부분 복셀 가중 및 관측 비율 채널은 프로젝트 추가 설계**, 직접 CP 근거 미확인 |
| 10–12 | 축별 CT 미분 | 3 | spacing을 반영한 수치 미분. 4 mm 배율 및 특징 채택 자체는 프로젝트 선택 |
| 13–15 | 축별 미분 계산 가능 여부 | 3 | 결측을 나타내기 위해 추가한 구현 메타데이터. 효과 미검증 |
| 16–18 | 상대 좌표 | 3 | 물리 좌표를 28 mm로 정규화. 이 스케일의 CP 근거 미확인 |
| 19 | 종양 signed distance | 1 | 물리 SDF/28 mm. 채택 효과 미검증 |
| 20 | 간 내부 깊이 | 1 | 물리 EDT/28 mm. 채택 효과 미검증 |
| 21–23 | 법선 | 3 | 물리 SDF gradient 정규화. 채택 효과 미검증 |

총 24차원이다. 여기서 x/y/z 이름은 array axis에 대응하며 해부학적 RAS 방향 보장을 의미하지 않는다. 반폭 4/12/28 mm는 구형 반경이 아니다.

r2는 계산 단위와 관측 마스크를 명시하려던 미완성 수정이다. 계산을 정교하게 만든 것과 연구 특징 선택을 검증한 것은 다르다. 이전 r1의 `16차원 + 패치 통계 20개` 설명 및 r1 DEBUG 성공 기록을 r2 검증으로 재사용할 수 없다. 이번에는 r2 테스트나 실제 CT/GPU 검증을 재실행하지 않았다.

## 입력보다 강한 가정: 복원 목표와 L2 teacher

| 항목 | 실제 코드 | 근거 대조 판정 |
|---|---|---|
| v2.1 context descriptor | `hiercp_v2/model.py:11`: 6종 node의 16차원 평균 96 + source/target의 5채널 평균·표준편차 20 = 116 | 입력 통계 요약을 학습 목표로 선택한 프로젝트 설계 |
| r2 context descriptor | `hiercp_v22/model.py:12`: 6종 node의 24차원 평균 = 144 | r1과 목표 차원이 달라짐. 물리 계산 개선만으로 볼 수 없는 objective 변경 |
| 복원 손실 | 두 model 파일의 `prompt_loss`: local embedding 및 label 기반 embedding으로 descriptor 복원 | 일반적인 복원 학습 선행 사례는 있으나 이 descriptor의 CP 유용성을 증명하지 않음 |
| L2 teacher | v2.1 164–172행, r2 166–174행: label assignment 가중 descriptor를 정규화하고, 다른 환자의 label 사이 cosine/temperature를 softmax한 분포에 correspondence를 맞춤 | 사용자 원문은 **정렬의 목적**을 제시했지만 이 **정렬 정답 생성법**은 지정하지 않음. 검토한 PRODIGY에도 없음. 현재 프로젝트가 추가한 미검증 가정 |
| T evidence와 L1 attention | v2.1 `_support_layer` 52행 이후 | attention 계산은 support embedding·label·owner로 구성. T/F는 data–label edge 속성으로 들어가지 않고 뒤의 positive evidence 집계에 쓰임. PRODIGY의 T/F edge attention과 같은 구현이라고 설명하면 안 됨 |

따라서 loss 감소나 backward 성공은 “이 teacher에 맞춰 학습할 수 있다”는 증거일 수 있으나, teacher가 올바른 환자 간 대응이라는 증거는 아니다. 입력 특징만 삭제하거나 교체하면 끝나는 문제로 결론내리지 않는다. 이번에는 대체 teacher나 새 특징을 선택하지 않았다.

## 이번 작업에서 정정한 주장과 확인 범위

- “수작업 특징에 아무 근거가 없다”는 포괄적 표현을 정정한다. 일반 선행 사례·계산 정의·첨부의 제안은 존재한다.
- “현재 16/24차원 조합이 유효하다”는 주장은 유지할 수 없다. 현재 CP 효과의 직접 근거는 확인되지 않았다.
- “전부 버리거나 특정 특징을 무조건 유지해야 한다”는 판단은 이번 근거에서 도출되지 않는다.
- “현재 L2는 PRODIGY 또는 사용자 원문의 구체적 학습식을 구현했다”는 주장은 성립하지 않는다.
- 정적 코드·원문·논문 대조를 수행했다. 새 단위 테스트, smoke test, 전체 학습, 전체 평가는 실행하지 않았다.
- 기존 r2 테스트/디버그 기록은 이번 논문 근거를 대신하지 않는다. 과거 실험 결과를 새 버전 성능으로 재분류하지 않았다.
- 핵심 소스/설정 9개 SHA256를 대조해 이번 문서 작업 전후 변경이 없는지 확인한다. 기존 code.txt를 새 r2 소스 export라고 주장하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 기존 문서는 원문을 남긴 추가 정정 방식이다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. 이번 근거 대조는 학습 실행이 아니므로 해당 검토를 새로 하지 않았다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 이번 작업에서 자원 측정 실험을 하지 않았다.
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번에 모델 실행/OOM은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다. 기존 DEBUG 증거를 성능 결과로 사용하지 않았다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이번에는 입력/손실 코드를 정적으로 추적했으며 backward/optimizer를 재실행하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 변경은 근거 기록·상태 정정뿐이다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 이번에 모두 미실행이다.
