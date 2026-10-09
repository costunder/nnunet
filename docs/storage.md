# 폴더와 파일 관리

버전과 실행은 [versions](../versions/README.md), 로컬 산출물은 [work 안내](../work/README.md)에서 찾습니다.
`기준선`은 비교에 쓰는 고정 파이프라인을 뜻합니다. 실제 CP 추천 품질이 모두 검증됐다는 뜻은 아닙니다.

| 찾는 것 | 위치 | 내용 |
| --- | --- | --- |
| 원본 V1 | [v1](../versions/v1/README.md) | 보존 소스, 기존 30mm 실행, 결과 출처 |
| 10mm 기준선 | [v1.4](../versions/v1.4/README.md) | 범위 축소 비교의 기준 |
| 현재 후보·loss 비교 | [selected](../versions/v1.8/selected/README.md), [native](../versions/v1.8/native/README.md), [fixed](../versions/v1.9/fixed/README.md), [listwise](../versions/v1.9/listwise/README.md) | 각 실험 설명, 독립 실행 입구, 체크포인트 위치 |
| 다른 모델 방법 | [v2.2](../versions/v2.2/README.md) | CNN, SAGE, 영역 축약, 희소 그래프 등 방법별 상태 |
| 실제 CT 원본 | `datasets/` | 학습 입력. 테스트 부산물과 구분 |
| 혈관 학습 입력 | `datasets/msd_liver/vessels/liver_1/` | 환자별 혈관 마스크·중심선·그래프. `vessels/index.json`에서 원본 CT·주석과 연결 |
| 혈관 시각화 | `work/vessels/liver_1/` | Slicer 장면·미리보기 |
| 보조 캐시 | `work/cache/` | 설치 파일, Slicer 임시 자료, 테스트 작업 공간 |
| 보관 자료 | `work/archive/` | 과거 혈관 결과, 테스트 출력, 이전 검토 자료 |
| 검증 증거 | `validation/`, `docs/` | 실행 결과와 진단 기록. 파일 이동 시 원문을 고치지 않음 |

## 이름과 저장 규칙

- 새 실험 결과는 `work/runs/<버전>/<방법>/<실험명>/`에 둡니다. 예: `work/runs/v1.9/listwise/seed42/`.
- 모델 설정·전체 commit·실행 시각은 실험 메타데이터에 기록합니다. 수정할 때마다 전체 hash와 긴 설명을 새 폴더 이름에 붙이지 않습니다.
- 같은 이름의 실험 결과를 덮어쓰지 않습니다. 반복 실험이면 `run01`, `run02`처럼 구분하고 각 폴더 README에 조건을 씁니다.
- 테스트가 만든 임시 자료는 `work/cache/tests/`에 생성합니다. 실제 CT/CUDA 검증 결과와 단위 테스트 자료를 혼동하지 않습니다.
- 기존 실험의 입력·source·checkpoint 경로는 저장된 계약을 유지합니다. 위 규칙 때문에 이미 진행 중인 실험을 새 경로에서 재시작하지 않습니다.
- 보관 파일은 삭제하지 않습니다. 이번 정리는 공간 확보 작업이 아니며 회수 용량은 0입니다.

## 기존 긴 경로를 남겨 둔 이유

`hiercp*`, `l0_*`는 서로 다른 구현을 import하는 실제 Python 패키지입니다. 패키지 이름, 소스 SHA, 실험의 절대 경로가 checkpoint 재개와 전처리 검증에 사용됩니다. 사람이 사용하는 입구는 `versions/`로 통일하고 이 구현 경로는 유지합니다.

`work`에 남아 있는 예전 DEBUG·source·cache 폴더 역시 [실험 위치 인덱스](../versions/artifacts.md)에 나옵니다. 실행기나 검증 기록에서 참조하는 폴더를 이름만 보고 옮기거나 삭제하지 않습니다. 모든 폴더를 최신·확정 결과로 취급하지 않습니다.

자동 생성 테스트 폴더는 실제로 `work/archive/tests/<분류>/<번호>/`로 이동합니다. 원래 이름과 새 위치, 파일 크기와 SHA256은 `work/archive/organization/tests-<정리시각>/`의 `plan.json`, `files.json`, `summary.json`에 기록합니다. `summary.json`이 있고 `sha256_and_size_verified=true`인 작업만 완료된 정리입니다. 과거 문서의 경로는 이 이동 기록으로 찾습니다.

로컬 정리는 서버의 `/home/.../Medical` 파일을 옮기거나 삭제하지 않습니다. Git으로 전달하는 것은 안내·실행 입구·관리 코드이며, 로컬 CT·캐시·체크포인트는 포함하지 않습니다.

## 작업 완료 체크리스트

이번 항목은 파일 관리에 한정합니다. 학습·자원 측정에 해당하지 않는 항목은 표시했습니다.

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
