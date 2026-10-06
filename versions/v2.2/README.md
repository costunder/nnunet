# v2.2

여러 L0 방법과 학습 수정이 같은 연구 계열 안에 있습니다. 실행 방법을 아래 폴더로 구분합니다.

| 폴더 | L0·학습 | 상태 |
| --- | --- | --- |
| [cnn](cnn/README.md) | 국소3D CNN | 기존 서버 순위 학습 약함 |
| [curriculum](curriculum/README.md) | CNN10mm, U16→128 누적 학습 | 구현/CUDA smoke 완료, 장기 결과 대기 |
| [sage](sage/README.md) | fine graph 유지 + SAGE | 기존 전환 경로 |
| [regions](regions/README.md) | 고정 partition + coarse SAGE | 영역 품질 미해결 |
| [sparse](sparse/README.md) | CNN 특징 + 작은 역할·문맥 그래프 | DEBUG, 추천 품질 미해결 |
| [explore](explore/README.md) | 연속 좌표 탐색 그래프 | DEBUG, 탐색 타당성 미해결 |
| [paired](paired/README.md) | v1 형태의 paired L0 | 과거 구현 |
| [early](early/README.md) | 초기 관측·관계 모델 | 과거 구현 |
| [control](control/README.md) | CNN/GraphUNet/SSN 대조 | 별도 연구·진단 |
| [history](history/README.md) | 변경 전 소스·검증 보존본 | byte-preserving archive |

각 폴더의 run.py가 실제 원래 실행기를 호출합니다. 기존 import package는 공유 runtime로 유지합니다.
`hiercp_v22 / hiercp_v221 / hiercp_v222`라는 파일 경로와 내부 format ID는 모델·artifact 검증 계약입니다.
사람용 버전/방법 폴더와 혼동하지 않습니다.

현재 선택한 새 실험은 **curriculum: 국소 CNN10mm + 누적 U16**입니다.
원래 D는 [v1.7/D](../v1.7/D/README.md)의 v1 그래프 L0 실험입니다.
모든128 후보를 평가한다는 사실만으로 두 모델을 같은 버전 설정이라고 보지 않습니다.

세부 소스는 [files.md](files.md), 결과 폴더는 [runs.md](runs.md)에서 찾습니다.

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
