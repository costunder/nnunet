# HierCP 연구 파이프라인

간 CT의 종양 Copy-Paste 위치를 학습·평가하는 연구 저장소입니다. **버전 선택과 실행은 [버전별 안내](versions/README.md)에서 시작합니다.** 버전 폴더는 사람이 찾기 위한 입구이며, 실제 모델·설정·데이터는 아래의 기존 경로를 사용합니다.

## 어떤 실험인가

| 구분 | 위치 | 현재 확인된 상태 |
| --- | --- | --- |
| 원본 V1, 30mm | [v1](versions/v1/README.md) | 예전 서버 checkpoint와 보존 소스를 구분한다. 원본30mm 전체 P+128U 최종 평가 결과는 아직 확인되지 않았다. |
| V1.4, 10mm 기준선 | [v1.4](versions/v1.4/README.md) | 사용자 제공 서버 기록상 40epoch 완료, 자체8후보 BEST MRR/top1=1.0. 공통 P+128U MRR=0.233839이며 별도 과제다. |
| 현재 네 군 비교 | [v1.8](versions/v1.8/README.md), [v1.9](versions/v1.9/README.md) | 원본 V1 10mm 모델로 좌표 노출·loss를 비교한다. 실제 CT/CUDA DEBUG와 재개 검증 완료; 네 군 전체40epoch 완료·최종 품질은 미확인이다. |
| A/B/C/D 원인 분리 | [A: v1.5](versions/v1.5/README.md), [B: v1.6](versions/v1.6/README.md), [C/D: v1.7](versions/v1.7/README.md) | A/B/C 자체8후보 학습 완료는 사용자 서버 기록에 근거한다. D의 최종 완료와 평가 수치는 미확인이다. |
| 별도 CNN·그래프 연구 | [v2.2 방법별 안내](versions/v2.2/README.md) | [CNN 누적 U16→128](versions/v2.2/curriculum/README.md)은 서버23epoch까지 보고되었으나 U16 gate를 통과하지 못했다. 다른 방법도 각 폴더의 검증 범위를 따른다. |
| 이전 방향·공유 기능 | [v2](versions/v2/README.md), [v2.1](versions/v2.1/README.md), [common](versions/common/README.md) | 폐기한 view-only 방향, 이전 환자 간 정렬, Basic CP·nnU-Net·feedback 공유 도구를 구분한다. |

원본30mm checkpoint의 소스 revision은 `a818158...`이며, 이후 비교의 기준인 [V1 보존 ZIP](versions/v1/pipeline_v1_source.zip)은 `74dcc2...` 시점이다. 현재 루트의 `hiercp/` 전체를 두 시점 모두와 동일한 원본으로 취급하지 않는다. 서버 요약으로 기록한 성능은 [버전별 results.json](versions/v1.4/results.json)의 출처와 독립 확인 여부를 함께 읽는다.

**8후보에서의 성공은 실제 CP 추천 품질 검증을 뜻하지 않는다.** 자체8후보는 원래 source anchor를 찾는 과제이고, 공통 P+128U는 donor 조건과 지표 정의도 다르다. P는 관측된 종양 위치, U는 미관측 비교 위치이며 CP 적합·부적합 정답이 아니다.

## 현재 네 군 실행·재개

| 군 | 학습의 비교 위치 | 목적함수 |
| --- | --- | --- |
| [selected](versions/v1.8/selected/README.md) | 원래 선별7개 | pairwise |
| [native](versions/v1.8/native/README.md) | 고정128U pool에서 매epoch 순환7개 | pairwise |
| [fixed](versions/v1.9/fixed/README.md) (`native_fixed`) | 같은 첫7개 반복 | pairwise |
| [listwise](versions/v1.9/listwise/README.md) (`native_listwise`) | native와 같은 순환7개 | listwise |

네 군 모두 학습 문제당 P+7개, 검증 문제당 전체129개와 원본 두 view를 유지한다. 각 군은 독립 checkpoint·optimizer·RNG·출력 폴더를 갖는다. [재개·캐시 안내](versions/v1.9/cache.md), [실행 비용 기록](versions/v1.9/runtime.md), [진행률·점수·병목 확인](docs/comparison_runtime.md)을 따른다. 문서에 이름이 있다는 사실만으로 현재 서버 프로세스가 실행 중이라고 판단하지 않는다.

```bash
python versions/run.py list
python versions/v1.9/run.py --help
```

버전별 `run.py`는 원래 실행기를 호출한다. 실제 GPU, 실험 root, resume, batch·worker 등은 해당 버전의 설명과 봉인된 실험 계약을 확인한다.

## 코드·데이터·기록 위치

| 경로 | 역할 |
| --- | --- |
| [versions/](versions/README.md) | 버전·방법별 README, 실행 wrapper, 설정·코드 링크, 결과 출처 |
| [hiercp/](hiercp), [hiercp_v1x/](hiercp_v1x) | V1 계열 runtime과 기준선·교차·네 군 비교 실험 연결 |
| [hiercp_v2/](hiercp_v2), [hiercp_v22/](hiercp_v22), [hiercp_v22_cnn/](hiercp_v22_cnn), [hiercp_v221/](hiercp_v221), [hiercp_v222/](hiercp_v222) | 이전·실험 계열의 실제 import package. 사람용 버전 번호와 일대일 대응하지 않는다. |
| [l0_local_cnn/](l0_local_cnn), [l0_regions/](l0_regions), 기타 `l0_*/` | 국소 CNN·영역·SAGE·sparse·exploration 방법 구현 |
| [config/](config), [tools/](tools), [tests/](tests) | 실제 설정, 실행·검증·서버 도구, 회귀 검사 |
| [basic_cp_online/](basic_cp_online), [custom_trainers/](custom_trainers) | Basic CP 및 nnU-Net trainer·feedback 공유 기능 |
| `datasets/` | 로컬 의료 원본·추출 데이터. 현재 `datasets/msd_liver/`가 존재하며 코드 배포물과 구분한다. |
| `work/vessels/`, `work/cache/`, `work/archive/` | 혈관 처리 산출물, 보조 캐시, 보존 산출물. 모델 소스 폴더가 아니다. |
| `work/<실험명>/`, `experiment_results/`, `output/`, `outputs/` | 실험 입력·checkpoint·캐시·생성 결과. DEBUG·UNIT·전체 실험 여부는 각 계약·보고서로 확인한다. |
| [validation/](validation), [docs/](docs), [실험 위치 인덱스](versions/artifacts.md) | 검증 증거, 설계·진단 기록, 기존 결과·캐시 위치 |

실제 Python package, 설정 원본, `versions/v1/pipeline_v1_source.zip`과 manifest, 실험의 `source/`·`shared/`·checkpoint·cache는 기존 경로와 바이트를 보존한다. import 이름뿐 아니라 소스 SHA, 절대경로, 입력·prototype·checkpoint identity가 재개 계약에 포함되므로 문서 정리를 이유로 이동하면 재개가 깨질 수 있다. 자세한 파일·실험 위치는 각 버전의 `files.md`·`runs.md`에서 찾는다.

2026-10-08 정리에서 자동 생성 테스트·과거 검토본 등 **503개 폴더, 37,440개 파일을 실제 보관 위치로 이동**하고 전체 파일의 SHA256·크기를 확인했다. 삭제한 파일은 없다. `work` 직속 폴더는 806개에서 305개로 줄었으며, 기존 실행·검증이 참조하는 경로는 유지한다. [로컬 파일 안내](work/README.md)와 [저장 규칙](docs/storage.md)에서 용도와 보관 기준을 확인한다.

이전 긴 루트 README는 [원문 보존본](docs/history/README_original.md)에 바이트 그대로 남겼다. 그 안의 “current/active” 표현과 상대경로는 작성 당시의 기록이다. 과거 변경은 [버전 이력](versions/history.md), 보존 소스는 [v2.2 history](versions/v2.2/history/README.md)에서 확인한다. 이번 파일 정리에서 새 학습·평가나 서버 파일 변경은 수행하지 않았다.

## 작업 완료 체크리스트

아래는 이번 문서 정리의 범위다. 새 학습·평가·자원 benchmark는 실행하지 않았다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 이전 README 원문을 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토한 기존 실행 계약·기록을 연결했다. 새 측정은 없다.
- [x] GPU, CPU, RAM 활용 상태를 측정한 기존 검증 기록을 연결했다. 새 측정은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사한 기존 기록을 연결했다. 이번 정리에는 OOM이 없다.
- [x] 디버그 설정과 최종 설정을 분리해서 안내했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결에 관한 기존 검증과 미검증 범위를 구분했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 이번 변경은 문서다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
