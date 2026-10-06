# V1.7 D

원래 V1의 L0·fixed48³/5채널·target erasure·6roles/3GAT·genuine 두 view와 native 독립 donor P+128U 과제를 결합한다. 현재 native LocalCNN V2.2 baseline과 L0가 다르다.

[실행](run.py) · [서버](server.sh) · [설정](config.md) · [진행 조회](status.py) · [공통 구성](../README.md) · [결과](../results.json)

서버 launcher의 `CP_ARM=D`를 선택하고 고정 `CP_OUTPUT`을 명시한다. physical32의 단위는 observations이며 실제 L0 작업은64개 sampled views다. 전체 train11279관측/validation2823관측과 support16 episode 학습, 전체 support 공통 평가를 유지한다.

기존 학습을 재시작하지 않고 진행을 읽으려면 [watcher](../../../tools/watch_v17_d_learning.py)에 실제 experiment를 지정한다. 저장된 MRR/Hit@1/R@1/pair-win 및 원래 BEST 선택을 표시한다. 최신 서버 전체40epoch 완료와 최종 성능은 완료 receipt·raw curve가 확인되기 전까지 미확인이다. 로컬 actual CT/CUDA2-update DEBUG는 별도 증거다.

마지막 사용자 보고의 output은 `/home/aicompetition06/Medical/experiments/v17_crossed_D_m10_seed42_20261005_reusefix`이며 epoch6, batch220/533, step2885였다. 현재 실시간 상태가 아니며 이 기록으로 최종 품질을 만들지 않는다.

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
