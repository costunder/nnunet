# D 학습 과정 표시

현재 D trainer는 실제 전체21 P+128U 검증을 수행하고 `training/curve.jsonl`에 초기(epoch0)와
완료한 각 epoch의 지표를 저장하지만, terminal에는 검증 진행률만 출력한다. update 진행
막대도 현재 batch의 loss/step/VRAM만 표시하므로 그것만으로 순위 학습 개선을 알 수 없다.

`tools/watch_v17_d_learning.py --experiment <D experiment> --follow`는 별도 터미널에서
기존 초기·모든 epoch 지표를 한 번에 출력한 뒤 새 epoch 지표를 이어서 표시한다.
현재 학습을 중단하거나 재시작하지 않는다. 기존 trainer/모델/loss/source identity를 바꾸지 않는다.
새 평가, checkpoint deserialization, GPU 선택, optimizer update, 원본 로그 수정은 하지 않는다.

출력은 다음과 같다.

- 초기 및 완료 epoch 전부: MRR, Hit@1, R@1, pair-win, P/U softplus loss, 저장된 NEW_BEST, epoch wall.
- 초기 → 최신 완료 epoch: MRR/Hit@1/pair-win의 실제 변화량.
- 현재 update: epoch/step, 최근 최대20 update의 평균 loss, 현재 rank/CE/alignment 항,
  clipping 전 gradient norm과 모듈별 gradient, update seconds, peak VRAM.

MRR은 환자별 최초 observed P의 reciprocal rank 평균이다. Hit@1은 평가 가능한 환자 중
최상위가 P인 비율이며, R@1은 전체 관측 P에 대한 micro recall로 서로 다른 지표다.
D BEST는 원래 계약인 `(MRR, R@1, -P/U_loss)` 선택의 `new_best` 기록을 그대로 따른다.
display 도구가 Hit@1이나 최고 MRR만으로 checkpoint를 다시 선택하지 않는다.

update loss는 타일/항의 가중치가 포함된 batch loss다. 감소나 gradient 존재만으로 held-out
ranking 개선이라고 판정하지 않는다. validation 지표로 변화량을 확인한다. P는 실제 관측
종양 위치, U는 미관측 비교 위치이며 CP 적합/부적합 label로 재해석하지 않는다.

조회는 모든 완성 JSONL 행을 읽는다. 최근20은 출력 평균의 window이고 데이터나 실험 subset이
아니다. 기록 누락은 경고하고 지표를 만들어 채우지 않는다. 작성 중인 마지막 줄만 pending으로
두며, newline까지 완성된 행의 파싱 오류·중복·뒤로 가는 cursor·파일 교체/축소는 명시적으로
거부한다. follow는 새 update 표시를 최대30초 간격으로 제한해 terminal 출력량을 줄인다.
새 validation 결과는 다음5초 polling에 바로 표시한다. 전체40 완료 receipt를 읽으면 끝나며
Ctrl+C는 별도 display만 중단한다. support refresh나 평가가 진행 중이면 새 검증 행은 아직 없다.

이 작업은 저장된 학습 과정의 표시 구현이다. 새 neural/GPU smoke, 전체 학습, 전체 평가를
실행하지 않는다. 실제 D의 서버 metric 값은 이 조회를 실행한 출력으로 확인해야 한다.

UNIT14/14 PASS: curve0..5 전체 표시, 다음 polling에서 새 epoch6 정확히1회 표시,
Hit@1/R@1 분리, 저장 BEST flag 준수, 부분 행 처리, malformed/교체/축소/소실 거부,
fresh-process torch/PyG/trainer 미import, 원본 journal 및 trainer SHA 보존,
checkpoint와 학습 완료 artifact 미생성을 확인했다. 이는 표시 동작 검사이며 새 학습 성능이 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 학습 설정 그대로 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 기존 update 기록에서 표시한다. 새로운 GPU 작업은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 작업은 OOM 변경이 아니다.
- [x] 디버그 설정과 최종 설정을 분리했다. UNIT JSONL 검사는 성능 증거가 아니다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 기존 핵심 모듈의 forward/loss/gradient/optimizer 연결은 유지하며 조회 도구는 그 경로를 변경하지 않는다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
