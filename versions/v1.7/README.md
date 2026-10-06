# V1.7 — C/D 교차 실험

A/B 대조 뒤 남은 **입력/L0 블록**과 **과제·학습 블록**을 교차한다. C/D 모두 B와 같은 native 상위 계층을 쓴다. 새 CNN baseline인 V2.2와 V1.7 D를 서로 같은 모델로 표시하지 않는다.

[C 실행](C/run.py) · [D 실행](D/run.py) · [설정](config.md) · [코드](code.md) · [결과](results.json)

서버: [C/server.sh](C/server.sh), [D/server.sh](D/server.sh). D 진행 조회: [D/status.py](D/status.py). 자세한 근거는 [전체 계약](../../docs/v17_crossed_training_20261005.md)에 있다.

| Arm | 입력/L0 | 과제·학습 | own-task 평가 |
| --- | --- | --- | --- |
| [C](C/README.md) | native raw crop, CT1채널, erasure 없음, CNN/organ mean/fused128, 1view | 원래 source-anchor, 선별8후보, AMP/cosine | 원래8후보 |
| [D](D/README.md) | 원래 V1 fixed48³/5채널, erasure, 6roles/3GAT, genuine2views | 독립 donor, 모든 P+128U live loss, FP32/constant LR, support16 | 모든 P+128U |

두 arm은 seed42/10mm/40epoch와 원래 전체 cohort를 유지한다. 원래 task corruption은 candidate datum과 묶여 있으므로 C/D에서는 off라는 coupling을 기록한다. 29축 장부를 모두 적은 것과29개 독립 binary contrast를 실행한 것은 다르다.

공통 평가는 동일21case의 P135+U2688=2823관측과 train P527+U10752=11279 full support를 사용한다. zero-P case도 전부 score한다. L0를 batch로 처리한 뒤 각 case의 모든 embedding을 모아 upper를1회 호출한다. query GT는 forward/prompt label에 주입하지 않는다.

C는 사용자 terminal 보고로 **40/40epoch·3040updates**, epoch 중앙값2.98분을 완료했다. own8후보 BEST epoch20은 MRR **0.986111**/top1 **0.972222**, 마지막 epoch40은 **0.918982**/**0.861111**이다.

| C의 P+128U 기록 | MRR | Hit@1 | pair-win | loss |
| --- | ---: | ---: | ---: | ---: |
| 학습 중 own8 BEST epoch20 | 0.195344 | 0.125000 | 0.563079 | 1.135099 |
| 학습 중 마지막 epoch40 | 0.108725 | 0.000000 | 0.563079 | 1.325501 |
| 이후 표준화한 공통 재평가 | 0.195354 | 0.125000 | 0.562500 | 1.131561 |

학습 중 최고 full128 MRR0.252844(epoch19)는 참고값이며 BEST 선택에 사용하지 않았다. 위 수치는 사용자 제공 요약이며 서버 checkpoint·raw JSON을 독립 읽은 결과가 아니다.

D의 마지막 사용자 보고는 `v17_crossed_D_m10_seed42_20261005_reusefix`에서 epoch6, batch220/533, step2885다. 이는 마지막으로 전달받은 기록이며 현재 실시간 상태나 최종 성능이 아니다. D 전체40epoch 완료·최종 품질은 아직 확인된 사용자 보고가 없다.

로컬 C/D actual CT/CUDA DEBUG와 resume 검사는 [실행 계약](../../docs/v17_crossed_training_20261005.md)에 보존한다. DEBUG 성공은 전체40epoch나 전체21case 품질 완료가 아니다. D의 현재 학습 지표는 [watcher](../../tools/watch_v17_d_learning.py)로 기존 로그만 읽어 확인한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 계약을 안내했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 기록만 참조하며 새 측정은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 문서 정리에 OOM은 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결에 관한 기존 증거를 구분했다. 새 학습 검사는 없다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 원본 실행 파일은 유지했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
