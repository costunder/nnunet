# v2.2 학습 상태 표시 패치

기존 tqdm에는 마지막 batch loss와 진행률만 있고 validation 수치는 파일에만 저장됐다. 화면에서 계산 진행과 학습 상태를 구분하도록 표시를 추가했다. 학습 구조·loss·support16·batch·VRAM 정책은 바꾸지 않았다.

## 화면과 해석

- `loss`: 현재 update의 전체 학습 loss. 표본과 환자가 달라지므로 매번 감소해야 하는 값이 아니다.
- `avg20`: 이 epoch/실행 구간에서 최근 최대 20 updates의 loss 산술평균. 재개 시 창만 초기화하며, 모델과 학습 진행을 초기화하지 않는다. epoch 전체 표본가중 평균이 아니다.
- `grad`: 기존 clipping 함수가 이미 계산하는 clipping 이전 전체 gradient norm.
- `probe=6/6`: CNN, SAGE1/2/3, L1/L2 각각 한 weight tensor의 최대 32개 고정 원소 중 optimizer 전후 변화가 확인된 모듈 수. 전체 parameter 갱신이나 정확도 향상의 보증이 아니다. `0/6`도 숨기지 않는다.
- `lr`, `sec`, `GiB`, `step`: 학습률, update 시간, peak GPU 할당, 완료 update.
- epoch validation 종료 시 ranking loss(낮을수록), MRR/Recall@K(높을수록), best loss와 `NEW_BEST`를 출력한다. 관측 종양 순위 지표이며 segmentation Dice나 CP 효용의 최종 평가가 아니다.

상세 loss 성분(rank/observation/alignment), 모듈별 probe delta와 tensor 이름은 `update_timing.jsonl`의 `learning`에 기록한다. probe는 RNG를 쓰지 않으며 작은 표본만 복사한다. 전체 모델을 매 update 복제/hash하는 감시 연산은 추가하지 않았다. 가중치 갱신 진단은 GPU 작은 gather/reduction 및 한 번의 scalar 묶음 CPU 읽기를 추가하므로 비용이 정확히 0은 아니다.

## 검사와 적용

11개 CPU/GPU 단위·회귀 검사가 통과했고 실제 CT DEBUG train8/val2, query2, 4 updates로 화면과 재개를 확인했다. 표시 전 8fadbd0 실행과 표시 후 실행의 최종 model/Adam/state/RNG hash가 모두 일치했다. 동기/비동기·중단/재개 비교도 일치했다. 실제 결과 원문은 `validation/region_learning_display_20260930/`. 이 검사는 전체 학습·전체 평가가 아니다.

기존 실행 화면은 실행 도중 자동 변경되지 않는다. Ctrl+C 한 번 후 PAUSED/RESULT를 확인하고 [A6000 재개 명령](region_execution_overlap_20260930.md#서버-적용)으로 최신 코드와 방금 저장한 체크포인트를 사용한다. `--resume-execution-upgrade`는 8fadbd0/10287dd의 검토된 실행 소스를 수용하며, 모델·Adam·RNG·epoch·step·batch·support를 보존한다. 이후 같은 코드에서는 일반 resume만 사용한다. 캐시 재생성이나 장기 학습 자동 시작은 없다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 정책 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 짧은 CUDA smoke와 기존 peak/RSS 로그 사용.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 수정은 표시 전용.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 단위 검사 합성 입력은 DEBUG로 구분.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
