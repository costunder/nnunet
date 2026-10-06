# 국소 CNN L0 입력 시각화

실제 donor `liver_1` component1 및 recipient `liver_66`의 두 기록을 사용했다.
전체 간 윤곽, 전체 원본 종양 주석, CNN 국소 입력 영역, 실제 native CT 단면을 표시한다.
노드·엣지·혈관·탐색 트리는 없다. 학습된 feature heatmap이나 예측 점수를 만들지 않았다.

Donor 실제 종양 bbox는 약18.922×13.516×15mm이다. 원래 donor anchor에 대한
bbox의 물리 offset을 recipient 후보로 옮기고, 각 면에5/10/15mm를 더한 범위를
비교한다.10mm는 시각화 초기 선택이며 production 범위를 확정한 값이 아니다.
Recipient의 종양 주석은 표시용이다. 후보 crop 크기 결정에는 사용하지 않는다.
간 외부 CT는 정규화 전에 제거한다. 외부를NaN으로 바꿔도 입력은 동일하고,
모든9개 preview 입력의 간 외부 값은0이다. native 해상도를 줄이지 않았다.
렌더링은 native voxel 중심과 cell face의 half-voxel 좌표 차이를 반영한다.

국소3D CNN은 voxel 배열에서 채널별 공간 feature map을 만든다. 기존
OrganPyramid의8 conv /12·24·32채널을 활용해10mm preview의 실제 CUDA forward
shape만 확인했다. Seed42 미학습 CNN이며, activation을 암 특징으로 표시하지 않았다.
전체 L0 readout/128D 결합부나 학습 파이프라인 완료를 이 시각화로 주장하지 않는다.
훈련 설정, 기존 학습 결과, Basic CP, L1/L2는 수정하지 않았다.

원시 기록: `work/paired_local_cnn_visual_20261001_r1/report.json`.
브라우저에서 donor/recipient 구분, CT 단면, 범위·후보 변경, 모바일 overflow 검사.

## 작업 완료 체크리스트

- [x] 서버/원격 세션 종료 위험 명령 없음.
- [x] 기존 파일·실험 결과를 파괴적으로 변경하지 않음.
- [x] 모델 깊이·너비 편의 축소 없음; 시각화만 생성.
- [x] 학습 그래프·데이터 규모 축소 없음; 실제 두 케이스 명시적 preview.
- [x] 숨겨진 학습 subset/cap/fast mode 없음.
- [x] 원본 읽기는4workers; CNN은 짧은 shape 검사이며 최종 batch 설정 아님.
- [x] GPU/CPU/RAM 확인 및 preview peak 기록.
- [x] OOM·모델 축소 없음.
- [x] 시각화/DEBUG와 최종 설정 구분.
- [x] 실제 CT와 원본 주석 사용; 가짜 예측 없음.
- [ ] Loss/optimizer 검증: 이번 요청은 시각화이며 학습을 실행하지 않음.
- [x] 범위 제안과 실제 구현 검사의 차이를 명시.
- [x] 전체 학습·평가 미실행 명시.
