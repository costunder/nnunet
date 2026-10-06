# 버전별 작업 폴더

현재 안내는 [START](../START.md)입니다. 긴 날짜·hash를 붙인 새 작업 폴더를 만들지 않고 아래 버전과 방법 이름을 사용합니다.

| 버전 | 실험 내용 | 안내 |
| --- | --- | --- |
| v1 | 기존 그래프, 원본30mm와 보존 source의 차이 | [열기](v1/README.md) |
| v1.4 | 10mm 기준선 | [열기](v1.4/README.md) |
| v1.5 | A: L0 교체 | [열기](v1.5/README.md) |
| v1.6 | B: 상위 계층·scorer 교체 | [열기](v1.6/README.md) |
| v1.7 | C/D 교차 실험 + 공통 전체128 평가 | [열기](v1.7/README.md) |
| v1.8 | 원본 10 mm 모델에서 비교 후보 조건 대조 | [열기](v1.8/README.md) |
| v2 | 폐기한 view-only 방향 | [열기](v2/README.md) |
| v2.1 | 환자 간 prompt 정렬 | [열기](v2.1/README.md) |
| v2.2 | 현재 방법별 구현 | [열기](v2.2/README.md) |
| common | Basic CP·nnU-Net·공유 기능 | [열기](common/README.md) |

각 폴더는 `README.md / run.py / config.md / code.md / results.json`처럼 짧게 읽습니다.
방법이 여러 개면 같은 버전 아래 `cnn / curriculum / sage / regions / sparse / explore`로 구분합니다.
폐기·미검증 방법은 상태를 표시하고 자동으로 학습시키지 않습니다.

Vx.yz에서 x는 방향 전환, y는 의미 있는 개선, z는 버그 수정입니다.
버그 패치마다 최상위 폴더를 만들지 않습니다. 과거 `v2.21`, `v2.22`라는 저장 이름과 내부 format ID는
[v2.2/history](v2.2/history/README.md)에 보존하며 새로운 의미의 버전 번호를 덧씌우지 않습니다.
v1.1~v1.3은 이전 계획에 있던 단계이며 실행 완료된 모델로 만들지 않습니다.

실제 실행 목록:

```bash
python versions/run.py list
python versions/v2.2/cnn/run.py --help
python versions/v1.7/C/run.py --help
```

원래 인자를 그대로 사용합니다. GPU·margin·출력·checkpoint를 자동 교체하지 않습니다.
실행 코드와 설정은 원본 파일명·SHA를 유지하고, 새 run.py가 그 파일을 실제 호출합니다.
각 버전의 [files.md](v2.2/files.md)에서 구현·설정·도구·검사·기록을 찾습니다.
[실험 위치](artifacts.md)는 기존 cache/checkpoint 경로를 버전별로 묶어 보여줍니다.
데이터 cache·결과·환경을 이름만 보고 더미라고 삭제하지 않습니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 설정을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 자원 기록을 연결했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 정리에서는 학습을 실행하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 구현 경로를 그대로 연결했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
