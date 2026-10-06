# v1.9 실제 CT·CUDA DEBUG 검증

[smoke.json](smoke.json)은 네 군의 실제 실행과 체크포인트 검증 결과다. [evidence_sources.json](evidence_sources.json)은 보존된 입력 계약·구현·검사 로그·실행 기록·체크포인트 84개 파일의 SHA256과 크기를 기록한다. Raw CT와 가중치 파일 내용은 이 폴더에 복사하지 않았다.

실제 train CT liver_5·liver_6과 별도 validation CT liver_31을 사용했다. 원본 v1 모델 10,434,532 parameters, L0/L1/L2 3/2/2층, 128D·4 heads·10 mm 범위를 유지했다. RTX 5070 Ti 16 GiB, PyTorch 2.8.0+cu128, PyG 2.6.1에서 실행했다. Physical source batch2는 16개 candidate graph와 32개 원본 sampled graph view이며 workers4를 사용했다. 서버 40 epochs 설정과 분리된 두 DEBUG epoch다.

각 군의 두 실제 update에서 전체 trainable tensor 1,085개에 유한한 gradient가 전달됐다. CNN·L0·L1·L2·scalar scorer의 표본 가중치가 모두 변경됐다. 초기 가중치 SHA는 네 군에서 같았고, 기존 v1.8 core 12개 파일과 원본 source/config/prototype/scope는 byte identity를 유지했다.

| 군 | 실제 학습한 고유 비교 위치/source | update peak VRAM | update 최대 RSS | update 중앙값 |
| --- | ---: | ---: | ---: | ---: |
| selected | 7 | 2.320 GiB | 13.968 GiB | 20.686초 |
| native | 14 | 2.178 GiB | 14.438 GiB | 19.587초 |
| native_fixed | 7 | 2.199 GiB | 14.649 GiB | 21.091초 |
| native_listwise | 14 | 2.175 GiB | 14.579 GiB | 21.088초 |

비용은 이 DEBUG source batch의 `step_seconds` 값으로, 전송·입력 검사·forward·backward·optimizer·checkpoint를 포함하고 loader 대기는 제외한다. 전체 평가를 포함한 서버 epoch 시간 또는 전체 cohort 메모리 상한으로 해석하지 않는다. 실제 graph당·relation별 node/edge 수와 loader·transfer·forward·backward·optimizer·checkpoint 시간은 smoke.json에 보존했다.

네 군 모두 초기·epoch1·epoch2 평가에서 원래 P 한 개와 동일한 U 128개 위치를 확인했다. L0만 chunk 처리하고 129개 후보를 한 upper graph에서 함께 점수화했다. Native와 listwise는 source·epoch별 학습 후보가 같았고, fixed는 두 epoch 모두 U0–6이었다. 실제 view epoch는 진행됐다. 이 두 DEBUG update로 최종 ranking 품질을 판단하지 않는다.

Listwise를 update1 뒤 중단해 optimizer·scaler·scheduler·RNG·source cursor를 저장한 후 정확히 재개했다. 네 군을 완료 후 다시 실행했을 때 추가 optimizer/backward/epoch completion은 모두 0이었고 모델·optimizer·scheduler·RNG가 보존됐다. 저장된 policy·checkpoint checksum도 대조했다.

새 UNIT 검사 31개와 기존 회귀 검사 74개가 통과했다. Native 접두사가 다른 군의 기록을 읽던 요약기 오류는 수정했으며 회귀 검사로 확인했다. Bash 실행기가 없는 로컬 환경의 shell syntax 검사 1건은 미검증이다. Linux/PyTorch2.6/CUDA11.8 서버의 자원 측정·전체 signed source 학습·40 epochs·최종 추천 품질·CP/nnU-Net 효과는 아직 검증하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 별도 DEBUG 입력임을 표시했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 실제 source2·32 graph views를 batching했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 DEBUG에는 OOM이 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
