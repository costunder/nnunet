# Basic CP80와 v2.22 CP80 비교

2026-09-22 사용자 요청에 따라 두 방법의 **CP 시도 확률을0.8**로 설정했다. 사용자가 말한 v2.2는 현재 작업 중인 v2.22 cluster-alignment r2 경로로 적용했다. 현재는 설정·구현·DEBUG 검증 완료 및 Basic 입력 준비 단계다. **양쪽 전체 학습·평가 결과는 아직 없다.**

## 비교 계약

| 항목 | 두 실행에서 공유 |
|---|---|
| CP 시도 확률 | 0.8 |
| seed | 42 |
| 데이터·분할 | 동일131CT; outertrain105/outerval26 |
| nnU-Net | ResEncM, 3d_fullres, patch128³ |
| native physical/effective batch | 2/2, accumulation1 |
| native 학습 | 250epoch |
| GNN 보조 학습 | v2.22에만 innertrain84/innerval21,40epoch |
| native worker | 양쪽 같은 실측값; 보정 아직 필요 |
| 평가 | 동일한 untouched outerval 및 기존 v5 평가 지표 |

기계 판독 설정은 `config/comparison_cp80.json`이다. Donor, 종양 크기 제한, 후보 위치, crop 정책의 기존 차이는 그대로이므로 **전체 방법 비교**다. GNN 점수만의 ablation 결과로 해석하지 않는다.

## 원본 보존과 실제 연결

- 기존 `nnUNetTrainer_250epochs_OriginalBasicCPOnline`은 확률1.0으로 보존했다. 원본 reference의 SHA와 CT/GT 동등성 검사도 통과했다.
- 새 `nnUNetTrainer_250epochs_BasicCP80Online`은 기본 donor/위치/크기/crop을 그대로 쓰고0.8 gate를 적용한다. `run_basic_cp_online.py train`의 기본은 승인된0.8 비교 트레이너이며, `--cp-probability 1.0`으로 보존 원본 트레이너를 명시할 수 있다.
- v2.22 config→bank metadata→catalog 검증→native trainer의 확률은 모두0.8이다. 실행 receipt에 실제 확률을 기록한다.
- 공통 epoch/worker/batch seed에서 event당5개의 난수를 사용하는 기존 v2 일정과 Basic gate를 연결했다. 첫 난수가0.8 미만이면 양쪽 모두 CP를 시도한다. 원본 Basic의 donor/위치 RNG는 별도이므로 그 소비량이 gate와 CT/기본 증강 일정을 밀지 않는다.
- 적용하지 않는20%는 원래 CT/GT를 반환한다. 시도했지만 유효 위치가 없으면 추가 재추첨 없이 원본을 반환한다. 따라서 **시도율과 실제 붙이기 성공률은 다르다**. Basic 방문 로그에 gate_u/attempted/status/확률/variant를 남긴다. v2는 native CP 적용 counter를 유지한다.

## 검증 및 현재 실행

DEBUG 검사66개 통과: CP80 신규6, Basic 원본8, 난수/native worker9, v2.2~v2.22 모델·누수/군집43.0.8 경계의 미적용, 실패 후 원본 반환, heldout 차단, 두 실제 loader의 시도 일정 일치, 원본1.0 보존을 확인했다. 최초 신규 test fixture에 hier_top_k가 빠진 오류를 수정한 뒤 통과했다. 정적 검사와 frozen-v1 hash도 확인했다.

수정 전 소스는 `versions/v2.22/before_cp80_comparison_20260922/`에 보존했다. Source hash가 변경됐으므로 이전 manifest/checkpoint를 새 hash로 덮어쓰지 않는다.

Basic 전체105CT의 immutable raw/native 입력 준비는 `work/cp80_comparison_20260922/basic_inputs/`에 시작했다. 명령과 wrapper PID는 같은 상위 폴더의 `basic_prepare.launch.json`, 진행/오류는 `.stdout.log`/`.stderr.log`에 기록한다. 이후 native 자원 보정과 실제 학습이 필요하다.

GNN은 `liver_108`의 현재 정답0 위치 규칙 때문에 보류 중이다. 현재 요구 거리27.9431mm, 최대 가능 거리23.5397mm여서 후보0개다. [비교 중심을 간의 종양 마스크 밖 위치로 정의하는 구체적인 제안](cp80_comparison_center_decision.md)에 대한 답변을 요청했으며 아직 적용하지 않았다.80% 요청을 별도 정답 정의 변경 승인으로 간주하지 않는다. Case 제외, 중심 수 축소, 가림 반경 축소는 하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch와 병렬화 계약을 검토했다. Batch2 유지, ordered worker 검사 통과.
- [x] GPU/CPU/RAM 확인: RTX5070Ti16GB,16logical cores,RAM64GiB; 시작 시 여유 VRAM14432MiB/RAM약51.5GB.
- [x] OOM으로 모델을 축소하지 않았다.
- [x] DEBUG 설정과 최종 설정을 분리했다.
- [x] 가짜 데이터·placeholder·random fallback을 production에 사용하지 않았다.
- [x] 핵심 모델 forward/loss/gradient/optimizer 회귀 검사와 실제 loader gate 연결을 확인했다.
- [x] 실제 적용 확률과 실행 범위 및 미해결 조건을 기록했다.
- [x] 검사·준비와 전체 학습·평가를 구분했다. 성능 비교는 아직 미실행이다.
