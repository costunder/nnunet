# v2.2 후속 서버 진단 실행

Singularity에 A6000 하나만 노출되어 `nvidia-smi -L`에서 GPU0으로 보이는 현재
터미널 기준이다. GPU 선택은 `CP_GPU` 하나로 한다. 호스트 터미널에서 실행한다면
현재 inventory의 사용할 물리 번호로 바꾼다. UUID를 하드코딩하지 않는다.
기존 진단용 checkout을 사용하며 production 학습 checkout과 출력은 보존한다.

아래는 학습 재개 명령이 아니다. 원래 학습 가중치/optimizer를 읽어 별도 진단만
실행한다. 새 scalar head만100회 진단 update하고 저장하지 않는다. FFN·attention·
history·alignment는 원본 상태를 변경하지 않는다. 전체 후보를 유지하는 기존
train2/validation2 case 선택 규칙과 checkpoint의 실제 physical batch를 사용한다.

```bash
CP_GPU=0
CP_DIAG_OUT="/home/aicompetition06/Medical/experiments/counterfactual_$(date +%Y%m%d_%H%M%S)"

cd /home/aicompetition06/Medical/HierCP-diagnosis-2ce11ba &&
git fetch origin codex/v222-server-r6 &&
git checkout --detach origin/codex/v222-server-r6 &&
python -u tools/diagnose_local_cnn_learning.py \
  --gpu "$CP_GPU" \
  --run /home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42 \
  --output "$CP_DIAG_OUT/report.json" \
  --cases-per-split 2 --workers 8 \
  --cuda-gib 24 --rss-gib 64 --resident-gib 24 \
  --deep --anchor-radius-mm 3 --message-scales 1 \
  --ff-scales 0 .25 .5 1 --history-only \
  --direct-head-steps 100 --direct-head-lr .0001
```

처음 명령은 빠른 원인 분리 경로다. `--message-scales 1`은 원래 baseline만
계산하며 이미 수행한 message4개 sweep을 다시 반복하지 않는다. 새로운
FFN4개와 attention 통계, history branch와 직접 head는 모두 실행한다.

**Fusion bypass는 위 명령에 `--fusion-scales 0 .25 .5 1`을 추가해 실행한다.**
이 옵션은 선택 case의 **전체 유효 train support**를 현재 CNN으로 한 번
재인코딩한다. 이를 빼고 새 query만 오래된 memory와 비교하는 지름길은 쓰지
않는다. CPU/입력·CNN 비용이 있으므로 이 단계까지 초단위 검사라고 부르지 않는다.
여러 λ가 CNN 결과를 공유하고 이미 trace한 query record도 재사용한다.
간 mask·모든 관측·후보·physical batch·L1/L2는 유지한다. 새 영구 CT/graph cache나
production checkpoint/ready 표시는 만들지 않는다.

현재 latest checkpoint가 로드되는 한 시점의 snapshot을 사용하며 시작 시
epoch 필드·step·hash를 표시한다. 서버 학습이 이후 latest 파일을 교체해도
다른 snapshot을 섞지 않는다. 이전 step11872를 반드시 재현하려면 실험 내부에
보존된 그 snapshot을 `--checkpoint`로 명시한다. 보존된 파일의 실제 경로를
모르면 임의 경로를 만들어 주거나 새 latest를 step11872라고 부르지 않는다.

요약에는 FFN별 gap/win/loss, attention sensitivity, Adam history 대조,
alignment tile 분포, 직접 head의 train/validation 전후 순위가 나온다. 전체
상세는 `report.json`에 보존하며 서버에서 JSON을 전부 cat할 필요는 없다.
저장된 결과를 다시 볼 때는 다음 reader만 실행한다. GPU 계산을 하지 않는다.

```bash
python tools/summarize_local_cnn_diagnosis.py "$CP_DIAG_OUT/report.json"
```

성능 개선 여부는 검증 순위와 원래 모델 대비 반응으로 판단한다. 분산이나
train loss만 커지고/줄어든 결과를 성공으로 올리지 않는다. 전체 nnU-Net Dice와
Basic CP 비교는 별도 최종 실험이며 이 명령으로 실행하지 않는다.

## 작업 완료 체크리스트

- [x] 세션/서버/부모 셸을 종료하는 명령을 넣지 않았다.
- [x] 기존 실험·가중치·cache를 덮어쓰지 않는다.
- [x] 원래 모델 규모·physical batch·모든 후보를 유지한다.
- [x] 진단 case/step 선택은 명시하고 production 설정을 변경하지 않는다.
- [x] CUDA/RSS/resident 예산과 병렬 reader를 명시한다.
- [x] Full support 재인코딩 비용과 짧은 head update를 구분한다.
- [x] Placeholder·무작위 예측·sample skip·fallback을 쓰지 않는다.
- [x] 진단 검사와 전체 학습/평가·성능 판정을 구분한다.
