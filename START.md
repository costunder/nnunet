# 시작 안내

[루트 README](README.md)에서 기준선·현재 비교 실험·코드와 데이터 위치를 확인한 뒤 [versions/](versions/README.md)에서 실행할 버전을 선택한다.

| 목적 | 안내 |
| --- | --- |
| 원본30mm와 10mm 기준선 구분 | [v1](versions/v1/README.md), [v1.4](versions/v1.4/README.md) |
| 원본 V1 10mm의 현재 네 군 비교 | [v1.8](versions/v1.8/README.md), [v1.9](versions/v1.9/README.md) |
| 기존 네 군의 재개·진행률·검증 점수 | [재개](versions/v1.9/cache.md), [실행 진단](docs/comparison_runtime.md) |
| A/B/C/D의 구조·학습 과제 비교 | [v1.5](versions/v1.5/README.md), [v1.6](versions/v1.6/README.md), [v1.7](versions/v1.7/README.md) |
| CNN·영역·SAGE 등 별도 연구 | [v2.2](versions/v2.2/README.md) |
| 기존 결과·캐시·공유 기능 | [실험 위치](versions/artifacts.md), [common](versions/common/README.md) |
| 파일 용도·현재 혈관 결과·보관 자료 | [로컬 파일 안내](work/README.md), [저장 규칙](docs/storage.md) |

버전 폴더의 README는 상태, `run.py`는 실행 진입점, `config.md`·`code.md`는 실제 구현 위치, `results.json`은 결과와 출처를 안내한다. 해당 파일이 있는 버전에서 사용한다.

8후보 학습 성공, 전체 P+128U 평가, CP 추천 품질은 서로 구분한다. 서버 완료가 사용자 기록에만 근거하는지, 실제 CT/CUDA DEBUG인지도 각 결과에 표시한다. 기존 소스·데이터·checkpoint는 원래 경로를 유지한다.

## 작업 완료 체크리스트

이번 변경은 안내 문서다. 새 학습·평가·자원 benchmark나 모델·데이터 이동은 수행하지 않았다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토한 기존 기록을 안내했다. 새 측정은 없다.
- [x] GPU, CPU, RAM 활용 상태를 확인한 기존 기록을 안내했다. 새 측정은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 조사한 기존 기록을 안내했다. 이번 정리에는 OOM이 없다.
- [x] 디버그 설정과 최종 설정을 분리해서 안내했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결에 관한 기존 증거를 안내했다. 새 실행 검사는 없다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
