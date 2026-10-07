# 보관 자료

| 폴더 | 용도 |
| --- | --- |
| tests/regions | 영역 그래프·서명·메모리·재개 단위 테스트 |
| tests/v1 | V1 실행 입구·범위·ROI·metadata 단위 테스트 |
| tests/transition | 교차 실험 전처리·병렬화·저장 단위 테스트 |
| reviews | 이전 GPT 검토용 묶음 |
| visuals | 이전 시각화 시도의 빈 작업 폴더 |
| tools/test-deps | 예전 격리 테스트용 의존성 사본 |
| vessels | 과거 혈관 결과·재검사·Slicer 장면 |
| organization | 이동 전후 파일 목록·SHA256·완료 기록 |

tests 아래 001, 002 등의 번호는 같은 종류의 테스트 실행을 구분합니다. 학습 버전·성능 순위가 아닙니다. 생성 당시의 긴 이름과 경로는 organization의 plan.json에 남습니다.

**학습용 원본 CT, 실제 모델 checkpoint를 더미로 분류하지 않습니다.** 이곳의 UNIT 산출물도 원본 바이트를 유지하며 삭제하지 않습니다. 실제 혈관 최신 결과는 [vessels](../vessels/README.md)에 있습니다.

정리 완료 여부는 각 summary.json의 sha256_and_size_verified로 확인합니다. 계획 파일만 있으면 아직 완료된 이동이 아닙니다.

## 작업 완료 체크리스트

이번 안내는 파일 관리에 한정합니다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. (파일 관리에 해당 없음)
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. (새 계산 실행 없음)
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. (해당 없음)
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. (모델 변경·새 검증 없음)
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
