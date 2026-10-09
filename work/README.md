# 로컬 데이터와 작업 결과

파이프라인을 고르려면 [버전 안내](../versions/README.md)를 엽니다. 이 폴더에는 코드와 구분한 데이터·결과·캐시를 둡니다.

| 폴더 | 무엇이 들어 있나 |
| --- | --- |
| [vessels](vessels/README.md) | 혈관 Slicer 장면·미리보기 |
| [혈관 학습 입력](../datasets/msd_liver/vessels/README.md) | 환자별 혈관 마스크·중심선·그래프. 현재 liver_1 한 사례 |
| [runs](runs/README.md) | 앞으로 만드는 실험의 짧은 버전/방법별 저장 위치 |
| [cache](cache/README.md) | 설치·추론·단위 테스트의 보조 파일 |
| runtime/totalseg | TotalSegmentator 실행 설정 |
| [archive](archive/README.md) | 예전 결과·검토 자료·테스트 출력. 삭제하지 않고 보존 |
| 그 밖의 기존 실험 폴더 | source·checkpoint·검증 기록이 기존 경로를 참조하는 자료. [버전별 목록](../versions/artifacts.md)에서 용도 확인 |

**혈관 그래프를 보려면 [vessels/liver_1](vessels/liver_1/README.md)만 열면 됩니다.** scene.mrb는 Slicer 장면이며, 그래프 데이터는 datasets/msd_liver/vessels에 있습니다.

자동 생성 UNIT 폴더는 archive/tests 아래 종류/번호로 옮깁니다. 이동이 끝난 기록은 archive/organization 안의 summary.json과 plan.json으로 확인합니다. 오래된 로그와 보고서에 적힌 경로는 원문을 고치지 않고 이 이동 기록으로 연결합니다.

서버 실험은 여기 있는 로컬 DEBUG와 별개입니다. 모델 가중치와 CT를 이름만 보고 더미로 판단하지 않습니다. 삭제는 수행하지 않습니다.

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
