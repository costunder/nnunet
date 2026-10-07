# 구현 파일

v1.9는 네 군의 조건과 실행 관리를 별도 entry로 연결한다. 원본 v1 모델 소스를 byte identity로 검증해 사용하며, 이 버전 폴더에는 원본 전체 소스 복사본을 만들지 않는다.

| 파일 | 역할 |
| --- | --- |
| [run_v19_comparison.py](../../tools/run_v19_comparison.py) | 네 군의 동일 초기값, 공통 batch calibration, 실행·재개 |
| [comparison_experiment.py](../../hiercp_v1x/comparison_experiment.py) | v1.9 계약과 네 군의 실제 공통 batch 선정 |
| [comparison_data.py](../../hiercp_v1x/comparison_data.py) | 실제 view epoch를 유지하며 fixed/rotating U 좌표 정책 연결 |
| [comparison_training.py](../../hiercp_v1x/comparison_training.py) | 네 군의 pairwise/listwise 목적함수와 보존된 학습 엔진 연결 |
| [v19_comparison_controls.json](../../config/v19_comparison_controls.json) | 네 군의 위치 규칙·목적함수와 공통 고정 조건 |
| [run.py](run.py) | 명시적 CLI 인자를 새 entry로 전달하는 버전 wrapper |
| [server_v19_comparison.sh](../../tools/server_v19_comparison.sh) | 호출당 추가 군 하나, 군별 독립 root·GPU·재개 |
| [v19_reference_execution.py](../../tools/v19_reference_execution.py) | 실행 중인 v1.8 JSON을 읽어 batch·worker 맞춤; 기존 파일 수정 없음 |
| [u_bridge_data.py](../../hiercp_v1x/u_bridge_data.py) | 보존된 original source/P/selected/frozen U 입력 기반 |
| [u_bridge_fields.py](../../hiercp_v1x/u_bridge_fields.py) | 정확한 whole-case distance 배열 저장·읽기 전용 mmap |
| [u_bridge_upper.py](../../hiercp_v1x/u_bridge_upper.py) | 원본 upper source/lesion 함수의 정적 출력 재사용 |
| [resume_comparison_cached.py](../../tools/resume_comparison_cached.py) | 이미 봉인된 네 군의 설정·최신 checkpoint를 유지한 재개 |
| [preparation_reuse.py](../../hiercp_v1x/preparation_reuse.py) | 다른 군의 완료된 동일 fields/upper/canonical 전처리 바이트 재사용 |
| [server_comparison_cached.sh](../../tools/server_comparison_cached.sh) | 호출당 기존 군 하나 재개, 기존 root 자동 확인 |

새 entry가 native_fixed의 첫 7개 고정 정책과 native_listwise의 CE 목적함수를 연결한다. 원본 graph 연산, P 메타데이터, source 목록과 sample 순서는 공통이다. 재사용되는 v1.8 파일 자체를 수정하거나 다른 군의 checkpoint를 초기값으로 바꾸지 않는다. 각 군은 자체 optimizer·scaler·RNG·cursor와 결과를 저장하고, 재실행 시 자기 최신 checkpoint에서 이어간다. [추가 캐시 경로](cache.md)는 네 군을 다시 시작하지 않고 이미 완성된 동일 전처리만 가져온다.

서버에서는 기존 v1.8 두 군을 재실행하지 않는다. 새 두 군은 서로 다른 root/data와 GPU로 동시에 실행할 수 있다. launcher는 v1.8의 완료된 calibration을 읽기만 하며 active 작업의 잠금을 가져오거나 해제하지 않는다. 초기 clone calibration을 제외한 장기 학습은 명시한 군 하나다. 요약기는 `--arms`로 해당 군만 표시한다.

Signed source 목록은 train 151개·validation 36개 문제이며 실제 validation은 18명이다. 설정 21명과의 차이는 실행 계약과 로그에 표시한다.

실제 입력 → 원본 graph → 모델 forward → 군별 loss → backward → optimizer update와 전체 129개 평가를 새 실제 CT/CUDA DEBUG에서 확인했다. 새 학습 facade는 원본 엔진의 동일 code object에 개별 정책을 연결하며, 기존 v1.8 파일 12개는 byte identity를 유지한다. Listwise의 중단·정확 재개와 네 군 완료 후 0 추가 update도 확인했다. [요약기](../../tools/summarize_v19_comparison.py)는 군 이름과 UUID를 정확히 구분해 네 군의 초기·best·마지막 full129 결과를 출력한다. [results.json](results.json)에 검증 범위를 구분해 기록하며, 서버 40 epochs와 추천 품질은 미검증이다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 네 군 실제 DEBUG calibration과 disjoint graph batch를 확인했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. v1.9 실제 자원과 구간 비용을 기록했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. DEBUG에서는 OOM 없이 원본 모델을 유지했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 전체 1,085개 tensor gradient와 주요 모듈 가중치 변화를 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
