# L0 수정안: CNN 고유 특징 위치와 특징 기반 연결

상태: **철회된 제안 — 구현 대상으로 확정하지 않는다.** 사용자 지적 후 이전 구조 자체도 감사하는 것으로 전환했다. r3 학습은 중단됐고 이 제안을 production에 적용하지 않았다. 아래는 당시 제안의 이력이며 현재 작업 지시가 아니다. [현재 감사](v222_graph_design_audit_20260923.md).

## 확인된 구현 문제

CNN은48³ CT에서32×12×12×12 특징맵을 만든다. `LocalEncoder.sample_features`는1728개 특징 위치를 삼선형 보간해45385개 노드로 늘리고, 이미지와 무관한6mm 반경 연결을 반복한다. 보간은 새로운 독립 영상 관측을 만들지 않는다. 다만 GAT의 특징별 attention과 비선형 변환이 있어 CNN과 수학적으로 동일하다고 단정할 수도 없다. 이 노드 구성의 추가 효용을 입증하지 않은 것이 문제다.

실제 그래프 입력은 outer_radius/node_spacing/edge_radius뿐이며 CT별 연결 변경은 없다. [읽기 전용 감사 결과](../work/v222_raw_ct_r3_20260923/l0_design_audit.json). CNN convolution 경로의 receptive field35는 GroupNorm의 공간 통계 의존성을 제외한 계산이며 전체 encoder의 엄격한 영향 범위로 해석하지 않는다.

## 구체적 수정 대상

1. 노드 하나를 CNN 원래 특징맵의 공간 셀 하나로 정의한다. 현 CNN 전체 map이면1728개다. CT 해상도와 CNN 깊이·채널을 줄이는 것이 아니다. 하지만45385노드의 보간 구형 영역을1728셀의 원래 cube map으로 바꾸므로 **노드 수와 공간 영역 계약 변경**임을 명시한다. 이전 결과와 같은 모델로 취급하지 않는다.
2. 현재 layer의 학습 특징을 기준으로 Euclidean kNN을 만들고 layer별로 갱신한다. 가까운 좌표만 연결하는 규칙을 대체해 영상 표현상 가까운 위치 사이에 정보를 전달한다. 특징 유사성을 종양/혈관의 실제 의미나 CP 적합성과 동일시하지 않는다.
3. CNN12/24/32, hidden128, L0 GATv2 3층·4heads, L1 data–label2층, L2 정렬2층, loss와 전체 데이터·split·seed42·CP80·epoch를 유지하는 별도 버전으로 구현해야 한다. 좌표/수작업 통계/GT 채널을 추가하지 않는다. discrete kNN 인덱스에 미분 가능하다고 주장하지 않으며 선택된 노드 메시지 경로를 통한 특징 gradient를 확인한다.
4. k를 논문 숫자로 임의 고정하지 않는다. 이웃 수별 관계 변화·feature collapse·처리량·VRAM과 inner-validation 결과를 확인한다. Test split은 선택에 사용하지 않는다. 후보 비교 자체와 최종 설정은 별도 기록한다.

노드 수·영역·연결 규칙 변경이므로 root AGENTS.md §2의 “사용자가 지정한 구조…기존 실험 계약에 정의된 값을 승인 없이 변경하지 않는다”에 해당한다. 현재 문서는 해당 변경을 구체적으로 검토할 수 있도록 만든 수정 명세다. production 기본값이나 실행 중 코드에는 적용하지 않았다.

## 근거와 차이

- Wang et al., **Dynamic Graph CNN for Learning on Point Clouds**, [원문 §3.2](https://arxiv.org/html/1801.07829#S3.SS2). 매 layer의 특징 공간에서 이웃을 재계산한다. 원래3D point cloud와 EdgeConv의 연구다. 본 CT/GATv2 구성을 직접 검증한 논문이 아니다.
- Han et al., **Vision GNN: An Image is Worth Graph of Nodes**, [원문](https://arxiv.org/abs/2206.00272). 영상 patch 특징을 노드로 쓰고 이웃 관계를 구성하는 선행연구다. 원문은 positional encoding과 고유 graph operator를 사용한다. CNN 특징만 유지하는 이 수정안은 그대로의 ViG 재현이 아니다.

## 검사와 미완료 항목

`tools/audit_v222_l0_design.py`에 원본 구조 감사와 독립적인 exact feature-kNN 진단 함수를 추가했다. 쿼리 chunk는 전체 노드를 순회하며 sample cap이 아니다. `tests/test_v222_l0_design_diagnostic.py`의4개 synthetic DEBUG가 특징 변경에 따른 연결 변경, 좌표순서와 무관한 feature 이웃, chunk 동등성·전체 노드 보존, 비정상 입력 거부를 확인했다. 이것을 실제 CT 모델 학습 또는 의료 성능 검증으로 보고하지 않는다.

남은 것: 승인된 새 계약의 production 구현, 전체 규모 batch 측정, GT가 forward에 들어가지 않는 검사, layer별 edge 변화율·연결 길이·노드 특징 붕괴 검사, gradient/optimizer 경로, 기존 고정 그래프 및 CNN-only L0와의 독립된 비교. L1/L2까지 제거하는 CNN-only 모델과 혼동하지 않는다. 어떤 방법이 더 좋은지는 아직 확인되지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 변경하지 않았고 변경 제안임을 명시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] 기존 physical batch 측정과 독립된 새 측정 필요성을 구분했다.
- [x] 현재 실행 로그를 확인했다. 새 모델 자원 측정은 미완료다.
- [x] OOM 회피를 위한 모델 축소를 적용하지 않았다.
- [x] 진단 DEBUG와 최종 설정을 분리했다.
- [x] 무작위 특징을 실제 학습 결과로 보고하지 않았다.
- [ ] 새 구조의 forward/loss/gradient/optimizer 연결: production 미구현.
- [x] 실행 중 baseline과 미적용 수정안을 구분했다.
- [x] 진단4개 통과와 전체 학습·평가를 구분했다.
