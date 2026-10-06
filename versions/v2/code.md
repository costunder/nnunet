# V2 구현 위치

V2 view-only 방향은 폐기됐으며 실행 소스를 보존 ZIP으로 확인한다. 이후 V2.1 수정 전 snapshot도 옛 V2 위치에 저장되어 있었으므로 아래에서 구분한다.

| 보존 코드 | 실제 archive | manifest |
| --- | --- | --- |
| 초기 view-only 원본 | [history/view-only/source.zip](history/view-only/source.zip) | [manifest](history/view-only/manifest.json) |
| V2.1 dtype 수정 전 | [history/dtype/source.zip](history/dtype/source.zip) | [manifest](history/dtype/manifest.json) |
| V2.1 shared donor 수정 전 | [history/donor/source.zip](history/donor/source.zip) | [manifest](history/donor/manifest.json) |
| V2.1 storage 수정 전 | [history/storage/source.zip](history/storage/source.zip) | [manifest](history/storage/manifest.json) |

[이력 안내](history/README.md) · [이동 장부](moves.json) · [SHA/ZIP CRC 증명](../../validation/version_layout/v2.json)

이동은 보존 경로12개에만 적용했다. 모든 payload bytes와 ZIP 내부 경로를 유지한다. 현재 실행 소스·설정·checkpoint·cache·학습 상태는 변경하지 않았다. 이 보존 폴더를 실행 모델이나 전체 학습 완료 증거로 사용하지 않는다.

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
