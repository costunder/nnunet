# GPU3 v2.3 BEST CP: 주변 context의 10mm 상한 복원

사용자는 10mm를 주변 조직 추출의 상한으로 지정했다. 기존 코드는 종양에서
2~10mm 거리이면서 간 마스크 경계에서 4mm보다 깊은 조직만 context로 선택했다.
`liver_46`의 첫 병변에서 2~10mm 안의 간 조직 2,334 voxel이 모두 깊이 3mm
이하라 제외됐고, 415/642개 채점 후 source context 필수 검증에서 중단됐다.
당시 nnUNet 학습은 시작되지 않았으며 기존 결과는 보존했다.

수정된 선택 규칙은 source와 target 모두 다음과 같다.

```python
context_mask = organ & ~footprint & (outside_tumor_mm <= 10.0)
```

2mm 하한과 간 깊이 제한을 선택 조건에서 제거했다. 실제 간 깊이는 기존 입력
feature로 계속 사용한다. 간 표면 역할은 원래 조건을 유지한다. context와
간 표면은 서로 다른 node type이므로 같은 물리 좌표를 가질 수 있다. 같은
역할 내부의 중복, 종양 내부, 비간 조직, 범위 밖 좌표는 계속 거부한다.
실제로 존재하는 간 표면 노드를 누락하면 오류를 내고, 실제 표면이 없는 경우의
기존 빈 surface 역할 허용도 유지한다. 주변 조직이 없는 경우 가짜 노드나
임의 embedding으로 대체하지 않는다.

변경은 `tools/v23_radius_context.py`가 GPU3의 분리된 CP 추론 runtime에만
적용한다. 보존된 v1 source, 다른 실험 runtime과 GNN 학습 checkpoint는
수정하지 않는다. 원래 schema의 양수 inner-radius 설정은 checkpoint의
과거 설정으로 보존하고, 실제 적용되는 inner=0 / outer=10 및 깊이 제한 없음은
별도의 `context_policy`에 기록한다. 이 정책은 요청, 파생 bank index와 완료
증명에 포함되며 다른 정책의 기존 점수는 재사용하지 않는다.

GNN은 선택된 v2.3 full128 validation BEST를 동결해 사용한다.
checkpoint SHA256은
`4d25477404d0c21b93b9fa03431665f07ad6df9c038e80aff85c7a6fb8f46c25`이다.
모델의 10,434,532 parameter, L0/L1/L2 깊이, hidden128, heads4,
기존 16개 graph feature, 5×48³ CNN 입력, 28mm normalization 및
shell4/12/28은 유지한다. context 입력 분포는 기존 GNN 학습과 달라진다.
이번 수정은 기존 BEST의 입력을 바로잡는 추론이며 GNN 재학습을 의미하지 않는다.

642개 기존 source payload와 source별 128개 후보 좌표, paste 규칙은 유지하고
score만 새로 계산한다. 기존 415개 점수를 보존한 채 새 출력 폴더에서 전체
82,176개 후보를 채점한다. 완료 후 원래 nnUNet 250 epochs, batch2,
patch128³, train105/validation26, CP 확률0.5를 실행하고 validation BEST로
원래 six-metric 평가를 수행하는 순서는 동일하다.

## 검증 상태

- 정적 검사 및 DEBUG 회귀 테스트 14개 통과.
- 실제 원본 `liver_46`의 mask/cache/depth/footprint alignment 일치 확인.
- 수정된 병변의 context는 실제 조직 2,808 voxel / canonical345 node이며,
  모든 좌표가 10mm 이내이다. 기존 나머지 세 canonical 역할 좌표는 유지됐다.
- 실제 전체 81명·642개 source admission 사전검증 통과.
  각 source의 주변 조직은 최소820 / 최대74,881 voxel이며 subset은 없다.
  증명: `/var/tmp/aicompetition06_v24_20261010/radius_all_source_preflight_DEBUG_1791700391620539665/report.json`.
- 실제 병변의 128개 후보 CUDA DEBUG에서 finite score와 동일 argmax,
  원본 payload 및 동결 모델 digest 보존을 검증했다.
  최신 증명: `/var/tmp/aicompetition06_v24_20261010/radius_context_actual_CT_DEBUG_1791700362382885037/report.json`.
- physical candidate batch32/64/128을 측정했다. 초기 실제 병변 검증의
  peak VRAM은 약1.45/2.29/4.00GiB였다. 최종 실행은 동일한 전체128 후보
  calibration으로 batch를 선택하고 실제 값을 별도로 기록한다.
- 서버 사용 범위는 GPU3, CPU affinity46~49의 4코어, RAM48GiB,
  GNN VRAM budget40GiB이다. 다른 사용자의 process/GPU는 변경하지 않는다.
- 전체 최종 채점·nnUNet 학습·최종 평가는 이 DEBUG 검증과 별개이다.
  프로덕션 실행 상태는 각 작업의 `status.json`과 `bank_complete.json`,
  `nnunet_complete.json`으로 확인한다.

수정된 전체 채점은
`/var/tmp/aicompetition06_v24_20261010/matched_cp_1791700482681236200/gpu3`
에서 RUNNING을 확인했다. 첫 source의 전체128 후보 채점이 완료됐다.
이 실행의 physical batch32/64/128 calibration은
11.82/11.22/11.95초, peak VRAM1.12/1.66/2.71GiB였고 같은 argmax를
유지해 batch64를 선택했다. 학습이 아닌 추론이므로 gradient accumulation은
없다. 아직 전체 채점·nnUNet 학습·최종 평가 완료를 의미하지 않는다.
MobaXterm 읽기 전용 보기 명령은
`/var/tmp/aicompetition06_v24_20261010/view_gpu3_cp_10mm.py`이다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (이번 오류는 OOM이 아니다.)
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (이번 모듈은 동결 BEST 추론 입력이며 forward를 검증했다. nnUNet 학습 경로 검증은 실제 학습 단계에서 수행한다.)
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
