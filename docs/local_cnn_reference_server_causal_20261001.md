# v2.2 — 실제 학습 대상·objective 방향의 서버 결과

이후 학습된 A6000의 **동일 pre-forward BN finite 결과**를 추가 수신했다.
최신 판단·이미 수행된 직접 head·v1과 현재 정답의 차이는
[2026-10-02 서버 기록](local_cnn_finite_server_20261002.md)에 연결한다.
아래 원래 결과와 당시 미검증 범위는 역사적 기록으로 보존한다.

사용자 `786be592` 첨부 전문을 보존했다. 실행 commit `70c8c5f`, 물리 GPU3
RTX A6000, 기존 m10 실험의 latest checkpoint를 읽은 독립 clone 진단이다.
이번 콘솔에는 snapshot epoch·step·hash가 없다. 이전 epoch39 결과와 같은
파일 이름이라는 이유로 가중치가 동일했다고 단정하지 않는다.

## 학습 대상 자체의 결과

원래 schedule4/5/6/7은 모두 liver_117이다. 각 tile은 P16+U16=physical32,
256쌍이다. Whole-case는 P71+U128=199개 전체와9,088쌍을 평가했다.
두 branch는 fresh AdamW로 시작하며 shadow는 각 step의 현재 clone moments를 복사한다.

| 전체 liver_117 | Legacy 전→4-update 후 | Reference 전→4-update 후 |
| --- | --- | --- |
| strict pair-win | .444542→.449824 | .477443→.448173 |
| 평균 P−U 점수차 | −5.40889e−8→−3.15076e−8 | −8.06184e−8→−1.53668e−7 |
| 표시된 평균 pair loss | .693147→.693147 | .693147→.693147 |
| 첫 양성 MRR | 1→1 | 1→1 |
| top5 양성 수 | 2→2 | 2→1 |
| top10 양성 수 | 3→4 | 2→2 |

학습 대상의 안정적인 개선이 확인되지 않았다. Reference strict win은2.927
percentage points 감소, legacy는0.5282 points 증가했으나 중간 timeline은 등락한다.
네 번의 clone update 결과이며 전체 학습 불가능성이나 통계적 유의성을 증명하지 않는다.

## MRR·동점·표시 정밀도

`tools/v22_rank_objective.py:110–122`의 MRR는 case별 첫 양성 rank의 역수를 평균한다.
모든 양성의 평균 reciprocal rank나 정확도가 아니다. liver_117의71개 양성 중
첫 양성이1위라 MRR=1이고 R@1=1/71=.0140845다. R@1의 최고값 자체도1/71,
R@5와 R@10 최고값은5/71,10/71이다. 작은 R@1만으로 실패를 주장하지 않는다.

Pair-win은 sP>sU인 엄격한 승률이다. 최초 `786be592` 콘솔에는 동점이 없었고,
아래 `02454e98`의 동일 JSON 재출력에서 실제 동점률을 확인했다. 동점 반점 비교는
`strict_win + .5*exact_tie_rate`다. 동점 순서는 score
내림차순 후 좌표·donor SHA256 순서이며 GT boost가 아니다.

최초 loss 표시는6 significant digits여서 ln(2) 근처의 작은 변화를 숨겼다.
이번12자리 재출력도 원래 FP32 계산의 정밀도를 높이지는 않는다. 전체 동점이나
FP32 quantization 지배를 확정하지 않는다.

## Objective·Adam·BN에서 확인한 것

Reference step2의 L1 weighted CE norm=32.2247, rank=8.22388,
rank↔CE cosine=−.837876, rank↔full=−.721407이다. L2도 각각−.793583,−.643796이다.
그 step의 목적 충돌은 실제 측정됐다. 그러나 step3/4 L1 rank↔CE는+.963019,+.904671이다.
모든 update에서 CE가 방해한다거나 CE 제거만으로 해결된다는 증거는 아니다.

모든8 full shadow delta가 actual full AdamW delta와 정확히 일치했다. 이는 계측·
optimizer 대조의 일치이며 ranking 개선 인증이 아니다. `cos delta rank-full`은
두 Adam delta의 cosine이다. 현재 rank gradient와 full descent의 cosine은 JSON의
별도 필드다. Moments 이후 두 delta가 비슷해지는 것만으로 유리한 방향이라고 판단하지 않는다.

첫 reference tile의 margin은 eval=3.57628e−7, BN-only=−1.37836e−7,
dropout-only=.00273491, current training=.0073373이다. Mode 민감성과 joint BN의
alignment→query 경로는 확인됐다. Dropout 출력 차이가 올바른 신호인지 잡음인지는 미확정이다.
BN-eval+dropout-off update의 margin=3.57628e−7→3.20375e−7,
win=.558594→.554688이므로 BN 고정만으로 해결된다는 증거도 없다.

Normalized Fisher는 초기 L0=.00152803, 이후 legacy L0=.001552,
reference L0=.00182413이다. Reference L1_2도.00160844→.00195415지만 순위는 개선되지 않았다.
평균 차이/분산을 prototype scoring의 유효 판별력으로 취급하지 않는다.
Train-only linear fit/held-out probe는 아직 미실행이다.

## 수정 대상을 좁히는 경계

이 결과로 GNN 종류·CNN 범위·채널·전체 데이터를 다시 바꿀 근거는 없다.
우선 대상은 현재 full objective의 이동이 최종 ranking 점수를 개선하는가다.
Rank-only/full shadow 이동을 동일 native tile·동일 BN snapshot·dropout-off·동일
frozen teacher에서 점수로 대조하면 auxiliary의 즉시 효과를 더 직접적으로 분리할 수 있다.
현재 shadow는 parameter 방향까지 저장하며 이 finite-step score 대조는 아직 실행하지 않았다.
Production CE/alignment weight나 모델을 자동 변경하지 않는다.

Whole-case는 full eligible saved support와 refit teacher, tile은 episodic support와
frozen teacher를 사용하므로 동일 조건으로 취급하지 않는다. 이번17–31s/iteration에는
추가 derivative·shadow 진단이 포함되며 production 속도 측정이 아니다.

증거는 `validation/reference_causal_server_20261001/server_console_original.txt`와
`server_console_transcription.json`이다.10 timeline·8 update·32 module rows·4 mode를
검증했고 표시되지 않은 값은 unavailable로 남겼다. 원본 서버 JSON은 로컬에 없다.
새 production checkpoint·ready·장기 학습은 만들지 않았다.

저장 JSON reader에 `--causal`을 추가했다. 재학습 없이 원래 저장된 동점·score 범위와
ranking gradient/full descent cosine을12 significant digits로 출력한다. 기본·signals
기존 출력은 byte-identical이며42개 출력·입출력 회귀검사를 통과했다. JSON reader는
표준 라이브러리만 사용한다. 이번 변경에서 새 모델 수치 검사를 CPU로 실행하지 않았다.

## 2026-10-02 동일 서버 JSON의 상세 재출력

첨부 `02454e98`는 `1e616d9`의 표준 라이브러리 reader가 위와 같은
`reference_causal_20261001_230955.json`을 다시 읽은 출력이다. 새 모델 실행이나
optimizer update가 아니다. 최초 원문과 전사는 그대로 두고 상세 원문과 전사를
`server_console_precise_original.txt`, `server_precise_transcription.json`으로 추가했다.
새 원문 SHA256은 `aa17f7314113bf5882cf96f90be071ee268340caf583c078459eccdf6c9629ba`다.
10 timeline·8 update·32 module rows의 결속과 동점 반점 계산을 검사했다.

| liver_117 전체199개 | Legacy 전→4-update 후 | Reference 전→4-update 후 |
| --- | --- | --- |
| 점수 범위의 최솟값 | −.772859931→−.571722746 | −.098206878→−.052211165 |
| 후보 점수 표준편차 | 7.19933e−7→7.99310e−7 | 2.02360e−6→1.30509e−6 |
| 후보 점수 max−min | 5.12600e−6→5.48363e−6 | 1.68085e−5→8.70228e−6 |
| 실제 동점률 | 4.1483%→3.2680% | 2.4538%→4.2254% |
| 동점에 .5를 주는 P/U 승률 | 46.5284%→46.6164% | 48.9712%→46.9300% |

Legacy의 최솟값·최댓값이 모두 약+.201137 이동한 반면 후보 간 차이는 약1e−6
수준이다. Reference도 약+.046 이동했고 표준편차는35.51% 줄었다. 후보 구분이
늘기보다 점수 범위가 공통으로 이동하는 현상이다. 공통 점수 이동은 P−U 순위
차이를 개선하지 않는다. 실제 동점은 일부이므로 완전 상수 출력이라고 부르지 않는다.
동점 반점 승률은 임상 정확도나 CP 효용이 아니며, 한 case의 짧은 clone 검사다.

새로 출력된 `ranking-gradient/full-descent`는 현재 train-mode rank gradient와
실제 full Adam parameter 이동의 **반대 방향**의 cosine이다. Legacy step4 CNN은
−.337932, fusion은−.082139이다. 같은 step의 두 Adam delta끼리의 cosine은
각각+.985019,+.989074여서, 두 optimizer delta가 비슷하다는 사실이 현재 ranking
loss를 내리는 방향이라는 뜻은 아님을 직접 확인했다. Reference step2 L1은
−.027626, L2는−.189753이다. 모듈별 first-order 방향이며 전체 loss 증가나
eval 점수 악화의 직접 증명으로 확대하지 않는다. 다른 update에서 CE와 rank가
같은 방향인 반례도 그대로 남긴다.

기존 tile-before와 tile-after는 같은 teacher·dropout-off이지만, reference의
native train forward가 중간에 BN running buffers를 갱신한다. 따라서 두 시점의
BN snapshot까지 같았던 대조로 설명하면 안 된다. 각 평가 함수가 원래 모델의
BN을 추가 변경하지 않는다는 보존 검사와는 다른 문제다.

이 출력에서 확인된 수정 대상은 점수를 분리하지 못하는 학습·scoring 경로다.
CNN 범위나 그래프 크기를 다시 바꾸거나, CE/alignment를 무조건 제거하는 패치는
이 결과만으로 정당화되지 않는다. 같은 BN snapshot과 frozen teacher에서
no-change/rank-only/full의 **실제 적용 후 native CT 점수**를 비교하는 통제가
필요하다. 이 수신 보고서에는 아직 그 finite-step 점수 대조가 없다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 서버32 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 서버 A6000 결과와 기존 GPU receipt 사용.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 서버 진단 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward/loss/gradient/optimizer 연결은 실제 서버8 update에서 확인됐다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 이번에는 수신 결과·출력만 수정했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
