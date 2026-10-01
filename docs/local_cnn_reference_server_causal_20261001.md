# v2.2 — 실제 학습 대상·objective 방향의 서버 결과

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

Pair-win은 sP>sU인 엄격한 승률이다. 동점은 별도로 세며 현재 콘솔에는 없다.
따라서 win<.5를 무작위 이하라고 해석하지 않는다. 동점 반점 비교는
`strict_win + .5*exact_tie_rate`이며 저장 JSON이 필요하다. 동점 순서는 score
내림차순 후 좌표·donor SHA256 순서이며 GT boost가 아니다.

Loss는6 significant digits로 표시된다. .693147은 ln(2) 근처의 작은 변화를 숨긴다.
콘솔만으로 정확히 같은 loss, 전체 동점이나 FP32 quantization 지배를 확정하지 않는다.
저장 JSON의 score std/min/max/tie/ranks는 원래 계산됐으며 재학습 없이 읽을 수 있다.

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
