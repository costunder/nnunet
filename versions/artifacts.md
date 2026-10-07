# 실험 저장 위치

[버전 안내](README.md)에서 실행 방법을 선택하고, 각 버전의 runs.md에서 기존 결과·cache·검증 위치를 찾습니다.

| 버전 | 위치·분류 항목 수 | 위치 |
| --- | ---: | --- |
| common | 62 | [runs.md](common/runs.md) |
| v1 | 36 | [runs.md](v1/runs.md) |
| v1.4 | 16 | [runs.md](v1.4/runs.md) |
| v1.5 | 5 | [runs.md](v1.5/runs.md) |
| v1.6 | 9 | [runs.md](v1.6/runs.md) |
| v1.7 | 38 | [runs.md](v1.7/runs.md) |
| v1.8 | 8 | [runs.md](v1.8/runs.md) |
| v1.9 | 13 | [runs.md](v1.9/runs.md) |
| v2.1 | 2 | [runs.md](v2.1/runs.md) |
| v2.2 | 253 | [runs.md](v2.2/runs.md) |

archive/tests는 완료된 SHA·크기 검증 receipt를 통해 묶음별로 표시합니다. 개별 원래 폴더명과 이동 위치는 원래→현재 링크에 보존하며, fixture 수를 학습 실험 수로 세지 않습니다.
cache·혈관 산출물·보존 폴더도 별도 항목입니다. 봉인된 runtime/checkpoint/cache 경로는 이동하지 않습니다.
테스트·DEBUG 이름만으로 더미라고 자동 삭제하지 않습니다. 이 인덱서는 데이터를 이동·삭제하지 않습니다.

새 실험은 버전/방법과 고유 실험 이름을 사용합니다. 기존 서버 실험 경로는 results.json 및 각 버전 README에 보존합니다.

## 작업 완료 체크리스트

이 파일은 탐색 인덱스다. 학습·평가·자원 benchmark와 데이터 이동·삭제는 실행하지 않는다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. (인덱스 생성에 해당 없음)
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. (인덱스 생성에 해당 없음)
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (인덱스 생성에 해당 없음)
- [x] 디버그 설정과 최종 설정을 분리해서 안내했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (새 실행 검사 없음)
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 기존 구현·기록으로 연결한다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
