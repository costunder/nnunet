# HierCP 버전 안내

**버전별 작업은 [versions](versions/README.md)에서 찾습니다.** 각 폴더의 이름과 역할을 아래처럼 고정했습니다.

| 폴더 | 내용 |
| --- | --- |
| [v1](versions/v1/README.md) | 원본 그래프와 기존30mm checkpoint |
| [v1.4](versions/v1.4/README.md) | 10mm 범위 축소 기준선 |
| [v1.5](versions/v1.5/README.md) | A: L0 교체 |
| [v1.6](versions/v1.6/README.md) | B: L1/L2·scorer 교체 |
| [v1.7](versions/v1.7/README.md) | C/D 교차 실험, 공통 P+128U 평가 |
| [v1.8](versions/v1.8/README.md) | 원본 v1 10mm, 기존7개 vs 고정128U에서 순환7개 대조 |
| [v1.9](versions/v1.9/README.md) | 같은 7개 고정·순환 및 pairwise·listwise loss 대조 |
| [v2](versions/v2/README.md) | 폐기한 view-only 방향 |
| [v2.1](versions/v2.1/README.md) | 환자 간 prompt 정렬 |
| [v2.2](versions/v2.2/README.md) | 현재 방법별 구현과 실험 |

`README.md`는 방법·상태, `run.py`는 실제 실행 진입점, `config.md`는 실제 설정 위치,
`code.md`는 구현 위치, `results.json`은 결과·출처를 담습니다. `files.md`는 세부 파일 목록입니다.
실행 파일은 원래 CLI로 인자를 그대로 전달합니다. 상대 입력·출력 경로는 저장소 루트를 기준으로 합니다.

```bash
python versions/run.py list
python versions/v1.7/D/run.py --help
python versions/v2.2/cnn/run.py --help
```

현재 추가 대조 실험은 [v1.9](versions/v1.9/README.md)의 **원본 v1 10 mm + 좌표 노출·loss 조건**입니다.
네 군 모두 source 원래 위치 1개를 정답으로 유지하고 전체 129개 후보로 평가합니다. [v1.8](versions/v1.8/README.md), 기존 [국소 CNN 누적 U16→128](versions/v2.2/curriculum/README.md)과 [D](versions/v1.7/D/README.md)는 별도 실험으로 보존합니다.

8후보 curriculum의 MRR/top1과 전체 P+128U의 MRR/Hit@1은 다른 평가입니다.
`P=실제 관측된 종양 위치`, `U=미관측 비교 위치`이며 CP 적합/부적합 정답으로 바꾸지 않습니다.

과거 스냅샷은 [history](versions/v2.2/history/README.md), 실험 파일은 [실험 위치](versions/artifacts.md),
공유 기능은 [common](versions/common/README.md)에서 찾습니다.
기존 import 경로·설정 원본·체크포인트·CT cache는 검증 결속을 유지합니다.
기존 루트 README는 보존된 v1/feedback 실행 설명입니다. 최신 버전 선택은 이 안내를 사용합니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. v1.9 네 군 DEBUG batch를 측정했고 서버는 공통 batch를 다시 측정한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. v1.9 실제 CT/CUDA 기록을 연결했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 DEBUG에서는 OOM 없이 원본 모델을 유지했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. v1.9 실제 update에서 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
