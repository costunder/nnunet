# V1 — 원본 30mm

V1은 원래 source anchor를 정답으로 삼는 8후보 curriculum 모델이다. 이후 10mm 분기와 A/B/C/D 실험은 [V1.4](../v1.4/README.md)부터 따로 안내한다. V1.1~V1.3은 계획 항목이며 실행한 버전으로 표시하지 않는다.

원본30mm 평가 진입점: [eval.py](eval.py) · [서버 eval.sh](eval.sh). 원본·기존 결과를 읽어 새 output에 평가한다.

| 구분 | 실제 소스·보존물 | 의미 |
| --- | --- | --- |
| 예전 서버의 원본 30mm | `a818158a81fc09b9d11d2774d53fba152bb2b453` | ROI30mm/context28mm의 기존 trained checkpoint 평가 |
| 이 폴더의 보존 ZIP | [pipeline_v1_source.zip](pipeline_v1_source.zip), [manifest.json](manifest.json) | `74dcc2cf03d2d40d1f582223321d96004333f661`, 202개 파일. 이후 10mm 비교의 보존 기준선 |

두 소스는 서로 다른 시점이다. ZIP의 내부 model revision v5를 예전 30mm checkpoint의 실제 소스라고 표시하면 안 된다. ZIP·manifest·checkpoint·cache와 기존 결과 경로는 원래 bytes와 식별자를 유지한다.

원본 30mm의 역사적 사용자 로그는 BEST epoch30 validation MRR **0.9869**, top1 **0.9804**를 보고했다. 당시 split105/26과 현재 공통 평가 split84/21은 다르다. 이번 정리에서 checkpoint를 독립 재평가하지 않았다.

| 사용 목적 | 진입점·기록 |
| --- | --- |
| 기존 파이프라인 진입점 | [run_v1.py](../../run_v1.py) → [run.py](../../run.py). 현재 root namespace의 진입점이며 예전 a818158 재현 소스와 동일하다고 보장하지 않는다. |
| 예전 30mm BEST의 공통 P+128U 평가 | [evaluate_native_v1_full128.py](../../tools/evaluate_native_v1_full128.py), [server_native_v1_full128.sh](../../tools/server_native_v1_full128.sh) |
| 평가 계약·DEBUG 증거 | [원본 30mm 평가 설명](../../docs/native_v1_30mm_full128_evaluation_20261006.md), [actual CT/CUDA DEBUG report](../../validation/native30_full128_DEBUG_20261006/report.json) |
| 10mm 기준선 | [V1.4](../v1.4/README.md). 공통 평가 표의 `V1`은 이 10mm 기준선이다. |

서버 원본 BEST는 `/home/aicompetition06/Medical/HierCP/work/full/model.pt`와 같은 폴더의 `prototype.pt`다. 평가기는 실제 original-source를 명시하여 original hierarchy/schema/prototype/region 연산을 로드한다. query GT는 forward 입력에 넣지 않는다. 외부 donor는 실제 donor/recipient 두 공간 frame으로 연결하며 recipient annotation을 사용하는 adapter다. 원래 단일 환자 topology의 exact replay나 annotation-free CP 품질로 표시하지 않는다.

현재 로컬 증거는 untrained actual CT/CUDA DEBUG다. full105 prototype bank를 유지하는 서버 전체21 P+128U 평가의 실제 수치는 새 서버 report로 확인해야 한다. 기존 bank와 현재 validation의 중첩을 별도 보고하며 독립 held-out 성능으로 주장하지 않는다.

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
