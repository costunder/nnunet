# HierCP 결과 및 검증 기록 정리 — 2026-09-18

현재 로컬 소스 `74dcc2cf03d2d40d1f582223321d96004333f661`과 함께 받은 문서·증거 ZIP을 정리했다. 서버에 접속하거나 학습·평가를 새로 실행한 보고서가 아니다. 기록의 최신 갱신일은 2026-09-16이며, 9월 18일 현재 서버 진행 상태를 뜻하지 않는다.

## 1. 다운로드된 자료의 범위

**GitHub 소스와 과거 검수 자료는 내려받았지만, 서버의 모든 실험 결과를 내려받은 상태는 아니다.**

| 자료 | 로컬 확인 | 의미 |
| --- | --- | --- |
| 모델·trainer·평가 코드, 설정, 테스트 소스 | 있음 | 실행 구현과 실험 계약 |
| `gpt_handoff.md`, `docs/` | 있음 | 과거 성능 수치, 수정·검증·서버 오류의 문서 기록 |
| `feedback/`의 보고서 3개·진단 코드·증거 ZIP | 있음 | 2026-09-12의 과거 코드 검수 |
| 증거 ZIP 내부 19개 항목 | 확인 | 보고서·진단 소스·목록·JSON·CSV·검사 로그·SHA 목록 |
| 실제 CT/정답 NIfTI | 없음 | 실제 의료 데이터 재평가 불가 |
| 학습 checkpoint·graph/population/online bank | 없음 | 원본 모델과 학습 상태 직접 검증 불가 |
| 실제 예측 마스크·환자별 평가 CSV·실험 `summary.json` | 없음 | 성능 수치 재계산 불가 |
| 최신 서버 journal·훈련 로그·실행 상태 | 없음 | 학습 완료/진행/중단 상태 직접 확정 불가 |

이번 확인에서 `.git`을 제외한 로컬 파일 전체를 조사했으며, 느슨한 파일 형태의 `.nii(.gz)`, `.pt`, `.pth`, `.ckpt`, `.npz`, `.npy`, `.csv`, `.log`, `summary.json`은 0개였다. ZIP 내부의 CSV/로그는 코드 감사 자료다. ZIP의 SHA 목록에 있는 18개 파일은 이번에 직접 검산해 모두 일치했다.

`README.md`는 의료영상·cache·checkpoint·생성 volume·nnU-Net 결과가 저장소에 없다고 명시한다. `.gitignore`도 `work/`, 결과 폴더, 의료영상, 모델과 로그를 제외한다. Git 업데이트가 서버 결과 동기화를 수행한 것은 아니다.

## 2. 문서에 남아 있는 실제 실험 성능

과거 사용자 제공 **fold 0 exact-argmax ablation**의 tumor Dice다. 이번에 원본 예측으로 다시 계산한 값은 아니다.

| 방법 | Tumor Dice | Basic 대비 절대 차이 | 해석 |
| --- | ---: | ---: | --- |
| Basic CP | 0.6454 | — | 과거 대조군 |
| Full M3 | 0.6722 | +0.0268 | Basic보다 수치상 높음 |
| w/o L1 | 0.6239 | -0.0215 | 이 비교에서 가장 낮음 |
| w/o L2 | 0.6977 | +0.0523 | 이 비교에서 가장 높음 |

차이 열은 기록된 반올림 Dice에서 산술 계산했다. 통계적 유의성이나 재현성을 추가로 입증하지 않는다.

- Full − w/o L2: **-0.0255**, 95% CI **[-0.0767, +0.0062]**, p **0.3241**로 기록돼 있다.
- 따라서 해당 단일 fold에서는 L2의 이득이 입증되지 않았다. L2가 불필요하거나 해롭다고 확정할 수는 없다.
- L1 제거의 낮은 점수도 단일 fold의 관찰이며, 이 자료에 L1 비교의 신뢰구간·p값은 제시되지 않았다.
- 별도 과거 `evaluation/summary.json`의 HierCP Dice **0.659247** 기록이 있다. 위 표와 다른 결과로 보존해야 한다. 해당 JSON 원본은 현재 로컬에 없다.
- F1/false-positive 개선에 대한 문서상 언급은 있지만, 확인한 인계 자료에는 이를 같은 조건으로 정리할 상세 수치가 없다. 추정해서 채우지 않았다.
- 위 결과를 현재 raw-target Feedback 또는 population-metric v5 결과로 재표기할 수 없다. 다중 fold 평균·표준편차나 최신 v5의 Full/Basic 성능표는 현재 자료에서 확인되지 않았다.

근거: `gpt_handoff.md:526`, `docs/online_evaluation_verification.md:102`.

## 3. 구현과 연구 성능을 구분한 현재 상태

현재 구현은 HierCP 2.2.0, graph schema `full_v22`, 상위 architecture `hiercp_source_content_population_metric_v5`다. L0/L1/L2, Quality GNN, 별도 Difficulty GNN, raw-target CP, nnU-Net 학습·평가 연결 코드가 포함돼 있다.

v4/v5 변경 기록에는 source-host 주소 연결 제거, 고정 0 입력 열 정리, population fit 증거·최종 군집 소속 보존, 관측 CT 기반 descriptor, prototype 거리 특징, 안전한 발행/재개, checkpoint에 연결된 평가가 포함된다. 이 요약은 기록된 변경 사항을 설명하며 이번에 전체 구현을 다시 감사한 것은 아니다.

유지된 설정은 H128, attention 4 heads, 3/2/2 blocks, 후보 128개, Quality GNN 40 epochs, nnU-Net 각 arm 250 epochs다. **설정에 적힌 epoch 수는 실제 학습 완료 증거가 아니다.** 과거 결과에 근거해 L2를 삭제하거나 모델 규모를 줄인 작업도 아니다.

현재 v5의 실제 전체 데이터 bank 완료, 40/250-epoch 학습 완료, 최종 paired 평가, 다중 fold 효과는 로컬 자료로 확인되지 않는다. 과거 checkpoint의 버전 이름만 바꿔 v5 결과로 사용할 수 없다.

근거: `README.md`, `gpt_handoff.md:98`, `gpt_handoff.md:138`.

## 4. 과거 검증 기록

아래는 서로 다른 날짜·revision·환경의 기록이다. 이번에 재실행한 결과가 아니며 테스트 개수를 합산하지 않는다.

| 날짜/대상 | 기록된 결과 | 범위와 한계 |
| --- | --- | --- |
| 9/12 독립 소스 감사, `ec2d838` | 159파일 목록, Python 134개·JSON 5개 검사, trainer SHA 10개 일치, 진단 8개 | ZIP 원시 증거 있음. 진단 성공에는 결함 재현 성공도 포함 |
| 같은 감사의 unittest 시도 | testsRun 405, success callback 344, error event 52, skip event 22 | 전체 실패. 이벤트는 서로 배타적인 분할이 아니므로 단순 합산 금지. 의존성·import 오류 등이 기록됨 |
| 9/12 추가 CPU 검수 | 선택 회귀 55개 + 별도 Difficulty 검사 1개 성공 | 제한된 CPU DEBUG 범위; 앞선 실패 기록을 지우거나 전체 suite 통과로 표현하지 않음 |
| 9/12 source-content v4 | 785개 중 780 통과, 5 skip | 당시 전체 CPU 회귀 기록. v5 검증으로 인용 불가 |
| 9/13 v5, 후속 수정 전 | 873개 중 868 통과, 5 skip | CPU DEBUG/회귀; 전체 의료 학습 아님 |
| 9/15 전처리 완료 증거 수정 후 | **884개 중 879 통과, 5 skip, 실패/오류 0** | 문서상 가장 최근 전체 CPU 회귀. Windows symlink 4개·opt-in 대규모 DEBUG 1개 skip |
| 9/16 Medical Data Aug 원본 통합 | **관련 회귀 77/77 통과** | 원본 연산·SHA·24 cell·resampling/trainer 등 관련 범위. 전체 884개는 재실행하지 않음 |

9/16에는 Python 166개 AST, JSON 5개 및 원본 source/출처 일치 검사도 기록돼 있다. 9/15 전체 suite와 9/16 선택 suite의 원본 실행 로그는 이 checkout에 없으므로 문서상 보고로 구분한다. 9/12 ZIP 안의 실패 로그를 최신 v5의 실패 결과로 간주해서도 안 된다.

raw resampling DEBUG에서는 native 대조 16건의 CT 최대 절대 오차 0, label/support 정확 일치가 보고돼 있다. 작은 합성 입력에 대한 정확성 검사이며 실제 환자 전체 데이터나 GPU 수치 동등성 검증은 아니다.

근거: `gpt_handoff.md:18`, `gpt_handoff.md:48`, `gpt_handoff.md:76`, `gpt_handoff.md:431`, `feedback/HierCP_code_review_20260912.md`, `feedback/HierCP_current_review_design_20260912.md`.

## 5. 기록된 서버 진행 및 중단 상태

서버 경로는 문서에 기록된 `/home/aicompetition06/Medical/HierCP-git`다. 아래 상태는 현재 서버 실시간 조회 결과가 아니다.

| 실험/단계 | 문서상 마지막 확인 | 확정할 수 없는 부분 |
| --- | --- | --- |
| `work/feedback_medical_aug` | 완료 Quality GNN/공통 native 전처리의 원본으로 기록됨 | 현재 파일 보존·SHA·재사용 자격을 이번에 직접 검증하지 않음 |
| 과거 bank 준비 | `liver_101` component 3의 1-voxel source가 원래 위치에서 resampling 후 소실되어 중단 | 이후 별도 실행의 성공 여부 |
| `work/feedback_rawcp` | `liver_15` component 1에서 저장 전 디스크 reserve 검사로 중단 | 전체 bank 완성과 Full/Basic 학습 완료 여부 |
| v5 시작 전 전처리 재사용 검증, 9/15 기록 | 완료 증거 4파일을 reader가 잘못 읽는 결함으로 새 root/runtime/journal 생성 전에 실패; 코드 수정 및 CPU 회귀 완료 | 수정 후 서버 재실행·v5 학습 성공 여부 |

디스크 실패 당시 free 약 **72.31 GiB**, 보존 reserve **80 GiB**, 해당 baseline 저장 하한 약 **1.03 GiB**, 누적 시간 약 **36시간 13분**, RSS 약 **290.6 GiB**로 기록됐다. 직접 중단 원인은 디스크 reserve였으며 OOM으로 기록된 것이 아니다. 후속 공유 NFS 여유 공간 341G 관찰은 전체 bank 완료를 보증하지 않는다.

오류 수정 코드가 저장소에 있다는 사실과 서버 실행이 끝났다는 사실은 구분해야 한다. 원본 bank/checkpoint/journal은 보존 대상으로 유지한다.

근거: `gpt_handoff.md:48`, `gpt_handoff.md:191`.

## 6. 전체 실험 결과를 완성하려면 필요한 원본

서버의 실험별로 아래 자료를 확보하면 성능표·학습 이력·완료 상태를 원본 기준으로 정리할 수 있다. 이 작업에서는 서버 다운로드나 새 실험을 실행하지 않았다.

1. 실험 이름, 코드 commit, 모델 버전, config, seed, fold/split, stage journal과 완료 receipt.
2. 실제 평가 `summary.json`, 환자별 지표, paired 통계, 평가 정의와 prediction/checkpoint 출처 정보.
3. 훈련 로그와 final/best/latest checkpoint 목록·SHA·저장 epoch; 필요 시 원본 checkpoint 자체.
4. bank index/manifest, 후보·source/cohort 수, 실패 기록, 전처리 완료 증거 및 resource preflight.
5. 결과 재계산까지 필요하면 해당 예측 마스크와 정답, 입력 cohort 대응 정보.

현재 자료로 가능한 결론은 **과거 fold 0 성능 비교가 기록돼 있고, 이후 v4/v5 구현·CPU 회귀가 진행됐지만, 최신 v5의 실제 서버 최종 성능은 아직 확인되지 않았다**는 것이다.

## 작업 완료 체크리스트

이번 작업은 결과 자료 정리다. 미체크 항목은 해당 연구 실행을 이번에 수행하거나 검증하지 않았다는 뜻이다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. 이번 자료 정리에서는 실행 설정을 변경하거나 측정하지 않았다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 과거 기록을 인용했으며 현재 자원 측정은 하지 않았다.
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 작업에서는 OOM이 발생하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다. 보고서에서 DEBUG 기록과 최종 실험을 구분했으며 설정 파일은 변경하지 않았다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이번 정리에서 재검증하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 원본 자료 조회·ZIP 18파일 SHA 검산·새 요약 문서 생성만 수행했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 이번에는 정적 코드 검사·단위 테스트·smoke test·전체 학습·전체 평가를 새로 실행하지 않았다.
