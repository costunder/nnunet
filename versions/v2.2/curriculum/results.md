# 23epoch 서버 결과

사용자가 제공한 서버 로그 전체를 읽었습니다. 결과 원본 checkpoint와 서버의 상세 JSON은 이번에 직접 읽지 않았습니다. 아래 숫자는 터미널 출력 정밀도이며, 모델·GT·gate·학습 설정은 변경하지 않았습니다. [수치 기록](results.json)

| 지표 | 초기 | 최고 validation epoch16 | 최근 epoch23 |
| --- | ---: | ---: | ---: |
| 전체 P+128U MRR | 0.117 | best 표시 0.2283 | 0.089 |
| 전체 P+128U observed R@1 | 0.000 | 0.022 | 0.000 |
| 전체 P+128U pairwise loss | 0.6932 | 0.6913 | 0.6930 |

23epoch 모두 query U16에 머물렀습니다. TRAIN gate pair-win 최고는60.58%, 최근55.02%여서70% 조건을 한 번도 통과하지 못했습니다. 최근 평균 bestP−bestU는+0.00922지만, 이 하나만으로 승급하지 않습니다. gradient와5/5 weight-change probe는 update 실행의 증거이며 유효한 순위 학습의 증거와 다릅니다. 최고 MRR의 일시적 상승은 있었지만 유지되지 않았습니다.

**이 결과는128개 query를 동시에 학습하는 것이 유일한 원인이라는 설명을 뒷받침하지 않습니다.** query만16U로 줄었고, 최적화 support에는 선택된16환자의 전체 P+128U가 들어갑니다. TRAIN gate와 validation은 전체 eligible support로 평가합니다. 최적화는 epoch 시작의 detached L0 bank, TRAIN gate는 epoch 마지막 update 뒤 refresh한 bank를 사용합니다. 그러므로 현재 gate는 최적화와 동일한 support 조건의 train-fit 측정이 아닙니다. 같은 checkpoint/query에서 episode support와 full support를 맞춰 비교해야 둘의 영향을 분리할 수 있습니다. [학습 경로](../../../l0_regions/training.py), [gate](../../../l0_regions/curriculum_training.py)

active counts/pairs/presentation multiplicity를 사용한 live ranking·CE 정규화에서는 새 오류를 찾지 못했습니다. auxiliary가 rank gradient를 얼마나 압도하거나 충돌시키는지는 이 로그의 total loss만으로 판정할 수 없습니다. 서버 `update_timing.jsonl`에는 loss별 값이 이미 저장됩니다. 큰 총loss를 pairwise ranking loss로 해석하면 안 됩니다. [loss](../../../l0_regions/donor_learning.py), [loss 기록](../../../l0_regions/learning_monitor.py)

epoch2–23의 optimization 중앙값은2분47.5초, 전체 support refresh는6분34초입니다. refresh는353batch로 그대로 남아 optimization보다 약2.35배 오래 걸립니다. query16만으로 전체 epoch 비용이 같은 비율로 줄지 않는 이유입니다.

별도의 실행 문제도 확인했습니다. native CNN loader는 매 요청 새 LocalBatch를 생성하지만 GPU cache는 그 객체 ID를 key로 씁니다. 반복 요청에서 같은 객체가 재사용되지 않으며, phase clear는 fine-graph 분기에서만 실행됩니다. 따라서 설정된8GiB cache에 재사용되지 않는 입력이 쌓일 수 있습니다. 로그의1.23→약8–10GiB 변화와 부합하지만, 서버 cache hit counter와 live tensor 통계를 읽은 것은 아니므로 autograd leak을 배제했다고 하지 않습니다. 이 cache 문제를 순위 학습 실패의 원인으로 연결하지 않습니다. [loader](../../../l0_local_cnn/data.py), [cache](../../../l0_regions/execution_pipeline.py)

기존 실행은 source/config identity를 checkpoint에 결속합니다. cache 구현·설정을 즉시 바꾸면 현재 실행의 exact resume가 거부될 수 있습니다. 이번 감사에서는 기존 학습·checkpoint·gate를 수정하지 않았습니다. gate를 낮추거나 다음16U를 강제로 추가해 성공으로 표시하지 않습니다. 현재 결과는23/40epoch 부분 기록이며, 전체128 curriculum 완료나 CP 추천 품질 검증이 아닙니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 batch32 및 support16 구현을 읽었다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 제공된 VRAM·시간 기록과 cache 구현을 확인했으며 서버 전체 자원은 새로 측정하지 않았다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 제공 로그에는 OOM이 없고 cache 재사용 문제를 코드에서 확인했다.
- [x] 디버그 설정과 최종 설정을 분리했다. 새 DEBUG/학습 실행은 없다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 코드 경로와 서버 probe 표시를 확인했으며 새 neural run은 없다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 제공된23epoch 부분 결과이며 전체40epoch 완료가 아니다.
