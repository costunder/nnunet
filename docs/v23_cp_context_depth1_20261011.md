# GPU3 CP context: 간 경계 깊이 1mm

사용자의 요청에 따라 source와 target의 context 선택 조건을 다음과 같이
변경했다. 간 마스크 경계 깊이는 **엄격하게 1mm 초과**이며, 정확히1mm는
제외한다. 종양 표면까지의 거리 상한은10mm이고 별도의2mm 하한은 없다.

```python
context_mask = organ & ~footprint & (outside_tumor_mm <= 10.0) & (organ_depth_mm > 1.0)
```

간 깊이는 기존 실제 feature로 유지한다. 간 표면 역할은 원래 조건을 유지하고,
서로 다른 역할이 같은 물리 좌표를 갖는 것은 허용한다. 역할 내 중복, 비간
조직, 종양 내부, 10mm 초과 또는 깊이1mm 이하 context 좌표는 거부한다.
실제 존재하는 간 표면을 누락할 수 없으며, 실제 표면이 없는 기존 허용 경로도
유지한다. 모델이나 데이터 규모를 줄이지 않았다.

직전 깊이 제한 없음 조건의 GPU3 채점 자식 PID1781219만 중단했다. 중단 전
소유자41833, 생성 시각1791700486.19, 전체 명령, 부모 PID1781218과 작업
manifest를 검증했다. 서버, SSH, 부모 셸과 다른 사용자의 작업은 건드리지
않았다. 완료된19개 source 점수와 기존415개 결과는 각각 원래 폴더에 보존됐다.
정책 변경 기록은 이전 작업의 `stopped_for_user_depth1_policy.json`이다.

새 정책은 `v23_matched_context_upper10mm_depth_gt1mm_v1`이며,
`effective_minimum_organ_depth_mm=1.0`과 엄격한 비교를 `context_policy`에
명시한다. 이 기록은 요청, 파생 bank와 완료 증명에 포함된다. 정책이 다른
이전 점수는 재사용하지 않는다. 원래 schema의 inner-radius 값은 과거
checkpoint 설정으로 보존하고, 실제 선택의 하한0은 별도로 기록한다.

전체642개 source와 source별128개 후보 좌표, 기존 CP payload와 paste 규칙,
selected v2.3 BEST와 모델 구조는 유지한다. BEST checkpoint SHA256은
`4d25477404d0c21b93b9fa03431665f07ad6df9c038e80aff85c7a6fb8f46c25`이다.
이번 실행은 수정된 실제 주변 조직을 입력으로 사용한 동결 BEST 추론이며,
기존 GNN 학습과 입력 분포가 달라진다. GNN 재학습이나 완료된 모델의 평가
결과를 의미하지 않는다.

## 검증

- DEBUG 회귀 테스트15개 통과. 실제 archived builder/validator로 깊이 정확히1
  제외, float32에서1보다 큰 다음 값 포함, 10mm 상한, 역할 및 좌표 검증을 확인했다.
- 실제 `liver_46` 첫 병변: eligible context1,104 voxel / canonical176 node,
  깊이1.4142~3mm, 종양 거리1~10mm. 기존 나머지 세 역할 좌표를 보존했다.
- 실제 후보128개 모두 새 조건의 context 존재 및 finite GPU score를 검증했다.
  원본 CP payload, 후보 좌표와 frozen BEST model digest는 일치했다.
- CUDA DEBUG physical batch64로 전체128개를14.88초에 채점했다.
  peak VRAM1.98GiB, RAM4.38GiB, CPU affinity46~49였다.
  보고서: `/var/tmp/aicompetition06_v24_20261010/depth1_context_actual_CT_DEBUG_1791701009836752494/report.json`.
- 전체 81명·642개 source admission 사전검증 통과. 모든 source에서 실제
  context가 존재했으며, source별 eligible voxel은 273~74,613개였다.
  CPU 4개, RAM 0.28GiB로 156.50초에 확인했다.
  10mm 후보 영역에 12mm+1voxel halo를 두어 깊이 >1mm의 선택 조건을
  확인했다. 최종 graph의 깊이 feature는 원래 전체-volume 깊이를 사용한다.
  보고서: `/var/tmp/aicompetition06_v24_20261010/depth1_all_source_preflight_DEBUG_1791701278157966330/report.json`.
  이 사전검증은 source의 입력 선택 검증이다. 전체 82,176개 target의 최종
  채점 완료나 nnUNet 학습 완료를 의미하지 않는다.

계산 범위는 GPU3, CPU4(affinity46~49), RAM48GiB 및 GNN VRAM40GiB이다.
새 최종 실행에서 전체 128개 후보를 유지한 채 physical batch 32/64/128을
새로 측정했다. 처리시간은 각각 11.45/11.40/13.37초, peak VRAM은
1.12/1.65/2.67GiB였으며, 측정 시간이 가장 낮은 64를 선택했다.
CPU 입력 준비 시간이 10.63~11.90초로 GPU 연산보다 길다. 이 측정은
첫 source의 결과이며, 모든 병변의 처리시간을 보장하지 않는다.

## 전체 실행

새 작업: `/var/tmp/aicompetition06_v24_20261010/matched_cp_1791701411038643800/gpu3`.
출력: `/var/tmp/aicompetition06_v24_20261010/experiment_gpu3_matched_cp_1791701411038643800`.
확인 시 `score_full_historical_bank`가 RUNNING이고, 642개 중 2개 source가
완료됐다. 채점 자식 PID 1789720의 소유자·생성 시각·전체 명령·GPU UUID와
CPU affinity를 확인했다. 배포된 context helper SHA256은 로컬 파일과
같은 `6ffc09b4b0fd7def4f71f8dfa574883e4ea989b4b7d12a1fd3c21f5db407da8f`이다.
실제 샘플에서 CPU 152.8~165.8%, RAM 약 5.25GiB, GPU utilization 0~52%가
측정됐다. CPU 준비와 GPU 채점이 교대로 진행되는 작업이다.

MobaXterm에서 진행 상황을 보는 명령:

```bash
/var/tmp/aicompetition06_v24_20261010/view_gpu3_cp_depth1.py
```

채점이 끝난 뒤 원래 nnUNet 250epochs/batch2/patch128³/train105/validation26/
CP0.5와 validation BEST의 six-metric 평가를 수행하도록 연결했다.
현재 전체 GNN 채점은 진행 중이고, 이번 조건의 nnUNet 전체 학습·전체
평가는 아직 실행되지 않았다. 이전 조건의 결과는 새 실험에 재사용하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (이번 변경은 사용자가 지정한 선택 조건이다.)
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (동결 BEST의 추론 입력으로서 실제 forward를 검증했다. nnUNet 학습 검증은 후속 단계이다.)
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
