# SSN 입력 단계에서 간 외부 제외

요청: 간 밖을 그래프에서만 삭제하지 말고 CNN이 간 밖 CT를 보지 못하게 한다.

구현 경로: `tools/ssn3d_organ.py` → `tools/run_ssn3d_organ_debug.py` → `tools/export_ssn_whole_context.py --organ-encoded`.

1. native CT에서 organ=(label 1 또는 2)를 적용한다. 간 밖 값은 보간 전에 제외한다. 경계 보간은 영상×mask를 보간한 뒤 보간 mask로 나누므로 간 밖 밝기가 섞이지 않는다.
2. 입력 평균/분산은 간 내부에서만 계산한다. CT와 XYZ 모두 외부 입력을 0으로 고정한다. 종양 종류/마스크를 추가 특징 채널로 넣지 않는다.
3. CNN 각 convolution block, batch-normalization 통계, pooling과 scale 결합에 mask를 적용한다. 기존 6×64채널, 학습 출력 15D, 최종 19D, 640419 parameter는 유지한다.
4. SSN 초기 중심 집계, 빈 중심 제외, 10회 배정, 재구성 loss를 간 내부로 제한한다. 외부 배정 질량은 0이다.
5. 유효 영역의 모든 연결성분을 보존한다. 모든 실제 면 접촉을 edge로 만들고, 노드 표시는 영역 내부 실제 voxel을 사용한다.

## 실제 GPU 검사

RTX 5070 Ti, physical/effective batch 2, 2×1×48×48×48, FP32, 32회 CT 재구성 **진단 update**. 8 GiB 한도, 측정 peak 2.322 GiB. 새 초기화이며 기존 32-update 진단과 동일 깊이·너비다. GNN/nnU-Net 장기 학습 및 production 경로 변경 없음.

- native 간 밖 CT를 ±100000 범위로 바꾸거나 NaN으로 바꿔도 재표본화된 입력 최대 차이 0.
- model-grid 간 밖 값을 변경해도 CNN 입력, CNN 특징, SSN soft assignment, hard partition, loss 차이 0.
- 모든 parameter gradient 최대 차이 8.1956387e-8. CUDA 역전파 비교 허용오차 atol=1e-6, rtol=1e-4 내 통과. **gradient까지 bitwise 동일하다고 주장하지 않는다.** 최초 실행은 bitwise gradient equality assertion에서 중단되었고 성공 결과로 취급하지 않았다.
- 간 밖 input gradient 0, 간 내부 input gradient nonzero. 모든 trainable parameter의 finite/nonzero gradient 및 CNN weight 변경 확인.
- 외부 feature 0, 외부 assignment mass 0.
- liver_66:1 내부 52812 voxel, 외부 0, 종양 330/330 보존.
- liver_66:19 내부 70882 voxel, 외부 0, 종양 288/288 보존.

## 미해결 사항과 표시 범위

새 hard partition은 각각 **4090노드/13254edge**, **4531노드/14183edge**이며 1-voxel 영역이 2121개와 2297개다. 32회 재구성 진단에서 영역이 많이 파편화되었다. 이를 숨기려고 작은 영역을 버리거나 합치거나 node/edge cap을 넣지 않았다. 외부 입력 차단 검증과 partition 품질 검증은 다르다. 이 결과는 의미 있는 종양 문맥 군집이나 최종 CP 성능을 입증하지 않는다.

여전히 두 국소 ROI 결과이며 전체 간 SSN 그래프가 아니다. 새 그림은 같은 liver_66의 전체 간·모든 종양 주석 위에 이 두 그래프를 실제 좌표로 배치한다. native 간·종양 파일과 기존 production cache는 그대로다. 예전 `ssn-liver-only.html`은 사후 clipping만 한 과거 결과이고, 이번 결과는 `ssn-organ-encoded.html`이다.

증거: `work/ssn_organ_encoded_DEBUG_20260930_r1/report.json`, 같은 폴더의 `.npz`와 `diagnostic_weights.pt`, `work/ssn_organ_encoded_visual_20260930/report.json`, `visual_checks.json`. 학습 상태는 debug-only로 저장한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 감소는 승인된 간 외부 제외에 한정했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] 기존 physical batch 2의 두 CT를 GPU에서 함께 처리했다.
- [x] GPU/VRAM, CPU, RAM을 확인하고 peak VRAM/RSS를 기록했다.
- [x] OOM 발생 없이 기존 크기를 유지했다.
- [x] DEBUG 결과와 production 설정을 분리했다.
- [x] 실제 CT를 사용하고 dummy 또는 random fallback을 사용하지 않았다.
- [x] CNN/SSN → loss → gradient → optimizer update를 확인했다.
- [x] 입력 변경·수치 검사·partition 파편화·미검증 범위를 기록했다.
- [x] 짧은 GPU 진단과 전체 학습·평가를 구분했다. 전체 학습·평가는 미실행.
