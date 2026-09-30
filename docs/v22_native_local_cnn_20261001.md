# v2.2 — 범위 지정형 국소 3D CNN L0

사용자가 국소 CNN L0로 진행하도록 승인했으며, 실행 시 L0 범위를 입력받도록 추가 요청했다. 기존 v1/SAGE/EZ-SP/exploration 구현과 결과를 보존하고 `tools/run_local_cnn.py`로 새 경로를 분리한다. 기존 GAT/SAGE checkpoint의 exact resume가 아니다.

## 입력과 모델

- `--margin-mm`는 필수다. donor 종양의 실제 occupied bbox에서 각 면 바깥으로 추가하는 물리적 거리다. 10은 10mm 정육면체나 지름이 아니다.
- donor bbox의 anchor-relative mm 범위를 recipient 후보 중심으로 이동한다. recipient 종양 GT로 crop 크기를 정하지 않는다. 전체 후보128·관측 양성·split·같은 donor/live 비교·L1/L2·loss·support episode·Basic CP 80%·원본 paste mask는 유지한다.
- CT native spacing/voxel을 유지하고 48³로 리샘플링하지 않는다. crop 크기는 donor와 spacing에 따라 달라진다. batch의 오른쪽 zero padding과 boolean organ mask로 묶으며, padding은 해상도 변경이 아니다.
- organ=(원본 label1 또는2). 간 밖 CT는 정규화 전 및 모든 CNN block에서 차단한다. recipient 종양/정상 구분은 CNN 입력 채널이 아니다. 원본 CT의 종양 영상 자체를 가리지는 않는다.
- 공유 encoder: Conv3D 8개(2/3/3), 채널12/24/32, stride1/2/4. 앞서 시각화한 encoder 크기를 유지한다. 각 scale의 organ-masked mean → concat68D → Linear128/LayerNorm/SiLU. donor/recipient/차이/곱 concat512D → MLP256→128D → 기존 L1/L2.
- L0의 graph node/edge, SAGE/GAT, EZ-SP, 탐색 offset head는 없다. L1/L2의 관계 학습은 유지한다.
- 현재 CNN은 fresh seed42로 시작한다. 기존 CNN 가중치를 다른 정규화 모델에 억지로 이식하지 않는다. 동일 범위/설정/데이터/소스의 새 CNN checkpoint만 exact resume할 수 있다.

## 반복 비용과 저장

준비는 전체 관측 목록의 같은 donor 배정과 메타데이터 저장이다. 그래프/partition 생성이 없다. raw CT는 hash와 물리적 grid를 검증해 병렬 로드하고 RAM 예산 안에서 재사용한다. crop도 bounded LRU로 재사용한다. 동일 batch의 동일 donor/동일 crop은 고유 index로 한 번만 CNN 계산한다. 학습 중 CNN feature를 영구 캐시하지 않는다.

CNN은 unique crop들을 동시에 처리하며 후보별 GPU forward 반복문이 없다. 다음 CPU batch는 한 batch ahead로 준비한다. pin_memory/nonblocking 전송, 선택 가능한 GPU input cache와 기존 순차 checkpoint writer를 사용한다. 모델/gradient를 축소해 메모리 초과를 숨기지 않는다. 범위가 커져 예산을 넘으면 명시적으로 실패한다.

범위는 inventory의 `local_cnn.margin_mm`, checkpoint identity 및 `local._extra_state`에 모두 저장한다. `--margin-mm`를 변경하고 이전 inventory/checkpoint를 넣으면 실패한다. 각 범위는 새 출력 폴더와 새 학습을 사용한다.

## 실행 예시 — 서버에서 사용자가 실행

아래 예시는 A6000의 기존 물리 GPU UUID를 유지한다. 이 코드가 서버 checkout에 반영된 뒤 사용한다. 이 작업에서 서버 학습이나 git push를 자동 실행하지 않았다.

```bash
CUDA_VISIBLE_DEVICES=GPU-73681bb7-5393-5774-9afb-99b5590083c9 \
python -u tools/run_local_cnn.py run \
  --cache /home/aicompetition06/Medical/HierCP-v22-e1e34bf/work/v22_full_prepare_20260928_logfix/paired_cache/index.json \
  --margin-mm 10 \
  --output work/v22_local_cnn_m10_run01 \
  --workers 16 --cuda-gib 40 --rss-gib 192 --resident-gib 128 \
  --batch-candidates 32 --support-patients 16 --device-cache-gib 8
```

5/20mm 실험은 `--margin-mm 5/20`과 각각 다른 `--output`을 지정한다. 다른 GPU를 할당받았다면 그 GPU UUID를 각 프로세스에 지정할 수 있다. 한 GPU의 여러 프로세스는 CUDA/RAM 예산의 합과 여유를 고려해야 한다. 자동으로 여러 장기 학습을 실행하지 않는다. 범위 외 조건을 고정해 비교하려면 physical batch 후보를 같은32로 지정한다.

Ctrl+C는 활성 update를 마치고 `PAUSED`를 출력하며 저장한다. 재개는 원래 inventory·같은 margin·같은 실행 옵션을 사용해 `train --cache <run>/inventory/index.json --resume <run>/training/checkpoint_latest.pt --output <new output>`로 실행한다. `run`은 새로운 실험 전용이다.

## 검증 범위

최종 검사: 단위/회귀 **35개 통과**, 정적 syntax10파일 및 diff whitespace 검사 통과. 모델 전체 **1,125,718 parameters**, L0 **293,332 parameters**다. 실제 CT DEBUG에서 CNN/readout/fusion/L1/L2가 모두 갱신되었다. 중단·새 프로세스 재개 model/Adam/state/RNG가 연속 실행과 exact 일치했다. 외부 NaN 교란의 출력 차이0, 간 밖 입력 gradient0이다.

같은 실제 pair의 padding 포함 unique3개 crop batch는 margin5에서 `[3,1,44,36,5]`, margin10에서 `[3,1,58,50,7]`, margin20에서 `[3,1,88,80,11]`이었다. 20mm는 별도 CLI prepare 및 inventory 저장도 확인했다. 실제128개 후보 목록에서 physical32/48/64 비용 경로를 실행했다. 모두 기존 DEBUG support를 사용한 짧은 측정이며 전체 코호트/최대 크기 보장이 아니다. 기존 회귀 검사가 끝난 뒤 비용만 재측정한 원시 결과는 `work/local_cnn_DEBUG_20261001_verify_r2/report.json`이다. epoch 속도로 외삽하지 않는다.

- synthetic 단위 검사는 실제 CT 성능 결과가 아니다. 간 밖 NaN 독립성·외부 gradient0·padding 불변성·양쪽 입력 gradient·입력 변조 거부·범위 값 검증·다른 범위 가중치 거부를 검사한다.
- 실제 CT DEBUG는 train8/validation2, physical2, 짧은 4update와 초기/epoch validation, refresh, best 및 final-memory/export를 검사한다. 전체 40epoch를 실행하지 않는다.
- 원시 실행 기록: `work/local_cnn_DEBUG_20261001_m10/train_r2/`. 첫 개발 중 실행 `train/`은 실행 중 소스 변경을 final export가 거부한 결과이며 정상 final artifact로 사용하지 않는다.
- 재개/CP/범위/physical batch 비용 검사: `tools/verify_local_cnn_debug.py`. `work/local_cnn_DEBUG_20261001_verify/report.json` 및 `_verify_r2/report.json` 통과. CP의 두 실제 후보를 점수화한 뒤 원본 full-mask로 관측 종양 위치를 제외하고 다른 유효 후보를 선택했다. 전체 온라인 nnU-Net 통합 학습 통과를 의미하지 않는다.
- 서버 전체128후보×전체코호트, 실제 nnU-Net 전체 학습, segmentation Dice 및 CP 효용은 미검증이다. DEBUG 순위 상승을 품질 증명으로 사용하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] CNN 깊이·너비는 앞서 시각화한 구조이며 임의 fallback 축소를 넣지 않았다.
- [x] 사용자 승인으로 L0 그래프를 CNN으로 교체했다. 전체 데이터와 후보 규모는 유지한다.
- [x] 숨겨진 subset/cap/fast mode를 추가하지 않았다. DEBUG는 별도 입력·출력이다.
- [x] physical batching, donor 중복 제거, 병렬 raw load와 prefetch를 구현했다.
- [x] RTX5070Ti 16GiB, CPU16 threads와 사용 가능 RAM을 확인했다.
- [x] 메모리 상한 초과 시 모델/범위 축소 대신 명시적 오류를 낸다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy/random fallback을 실제 결과로 사용하지 않았다.
- [x] CNN/readout/fusion/L1/L2의 loss·gradient·optimizer 연결을 실제 CT DEBUG에서 확인했다.
- [x] 실행 범위와 변경 사항을 기록했다.
- [x] smoke와 전체 학습·평가를 구분했다. 전체 학습·평가는 미실행이다.
