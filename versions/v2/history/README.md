# V2 보존 이력

이 폴더는 **폐기된 V2 view-only 원본**과 이후 **V2.1 수정 전 snapshot**을 보존한다. 모두 V2라는 옛 저장 위치에 있었던 기록이며 같은 모델 버전으로 묶어 해석하지 않는다. 현재 실행에는 이 ZIP을 사용하지 않는다.

| 폴더 | 실제 이력 | 보존 소스 |
| --- | --- | --- |
| view-only | 동일 데이터 두 view 정렬. 잘못된 L1 task identity/view-only L2로 폐기된 초기 방향 | [source.zip](view-only/source.zip), [manifest.json](view-only/manifest.json), 218개 파일 |
| dtype | V2.1의 CUDA BF16 index_add dtype 수정 전 | [source.zip](dtype/source.zip), [manifest.json](dtype/manifest.json), 2개 파일 |
| donor | V2.1의 shared donor/누수 방지 수정 전 | [source.zip](donor/source.zip), [manifest.json](donor/manifest.json), 17개 파일 |
| storage | V2.1의 lossless source deduplication/lazy lesion mask 수정 전 | [source.zip](storage/source.zip), [manifest.json](storage/manifest.json), 5개 파일 |

[검증 기록](../evidence/)에는 checks, cross-patient, cross-patient-final, cuda, handoff, patches, donor, storage를 나누어 두었다. 날짜·format ID·당시 경로가 적힌 원본 기록 내용은 그대로 유지한다.

[이동 장부](../moves.json)와 [검증 증명](../../../validation/version_layout/v2.json)은 이번12개 폴더 이동만 기록한다. 41개 원본 파일의 이동 전후 SHA, ZIP5개의 CRC, source manifest4개가 지정한242개 내부 파일 SHA를 확인했다. 루트 moves.json의 이전34개 이동 증명과 당시 V2 pinned 기록은 수정하지 않았다.

현재 runtime이 실제 결속하는 [V2.1 이전 소스](../../v2.1/before_v22_20260920/manifest.json)는 별도 경로에 유지한다. 여기의 짧은 이름은 찾기 위한 안내이며 source/checkpoint 식별자 변경이 아니다.

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

