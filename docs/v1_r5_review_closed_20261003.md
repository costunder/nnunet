# r5 독립 검토 마감과 다음 순위 학습 대조

## 판정과 근거

사용자가 전달한 독립 검토는 실제 r5 ZIP을 해제하여 ZIP 직접 입력, directory 입력, 세 artifact 검사 파일을 재실행했다. runtime416 8개, nested 결속 6개, bundle-input 14개로 총 28개 metadata/contract regression이 PASS했다고 보고했다. 로컬에 보존된 독립 실행 receipt와 같은 ZIP을 검토한 결과다.

- ZIP: `exports/HierCP_v1_STRICT_NESTED_GPT_REVIEW_20261003_r5.zip`
- SHA256: `ef5814d011889dc2ad14c453180c33f63eb0df7ae4d8e61fb29b1a75b5af418e`
- 외부 ZIP 99 members / manifest payload 98 files, CRC 및 payload SHA/size 검사 PASS.
- 보존 원본 v1 source 202개 파일 일치, runtime416 계약·초기 neural state·10,434,532 parameters 결속 유지.
- 로컬 기존 receipt: `validation/v1x_progressive_20261003/bundle_standalone_verification.json`.

이번 기록 시점에는 로컬 ZIP SHA·멤버 수와 기존 receipt의 일치만 다시 확인했다. 외부의 28개 재실행과 이전 로컬 28개를 합쳐 더 큰 검사 수로 보고하지 않는다. 새로운 GPU run, 학습, 평가 또는 ZIP 생성은 수행하지 않았다.

Artifact·standalone 재현성·실행 계약 검토는 마감한다. r5를 그대로 보존하며 새로운 packaging/debug 패치를 추가하지 않는다. sampler와 모델도 고정한다. r5는 전달본 개정 번호이며 새 모델 버전이나 품질 통과 표시가 아니다.

## 고정한 첫 비교

| 항목 | native arm | strict-nested416 arm |
| --- | --- | --- |
| 모델 stage | 원본 v1.0 | 원본 v1.0 |
| 정답·학습 목표 | source 실제 anchor와 원본 curriculum/corruption ranking | 동일 |
| 모델·loss·CNN 범위 | 원본 그대로 | 동일 |
| 데이터 분할 | train 84 / validation 21 / outer 제외 26 | 동일 |
| 학습 | seed42, 40 epochs | 동일 |
| 후보 | pool128에서 원본 curriculum의 8개/sample, 두 view | 동일 |
| canonical preparation/prototype | native reference가 한 번 준비 | 같은 native reference 공유 |
| physical batch / worker | 원본 preflight에서 측정 | 같은 측정 lock 적용 |
| local sampled view | 원본 native view | 그 native view 내부 strict subset 및 induced edges |
| 역할 seed 예산 | 원본 규칙 | `64 / 32 / 96 / 64 / 96 / 64` |

416은 seed 예산의 합이며 최종 node cap, 검증된 최적값 또는 품질 admission이 아니다. 같은 seed와 초기 neural weights가 서로 다른 graph shape에서 동일 dropout/optimizer trajectory를 보장하지 않는다. v2.2 observed P/U 정답이나 CP suitability 정의를 이 v1 고정-GT 비교에 섞지 않는다.

## 실행 경로와 판단 순서

현재 구현된 경로는 `tools/run_v1x_experiment.py`의 명시적 native init, native prepare, native train, native reference를 지정한 nested416 init/train이다. nested init은 native 학습 전에도 가능하지만 nested train에는 native의 측정된 실행 lock이 필요하다. 준비는 native에서 한 번 수행하며 기존 v2.2 paired cache를 자동 채택하지 않는다.

`init`과 `plan`은 학습을 시작하지 않는다. 실제 학습은 `run --stage v1.0 --target train --gpu <해당 환경의 번호>`로 별도 시작한다. 두 arm은 별도 results/checkpoint 경로를 사용한다. 원본 v1은 마지막으로 저장한 완결 epoch에서 재개하며 중간 batch가 저장됐다고 주장하지 않는다. 서버 배포·Git push·서버 학습은 이번 검토 마감 작업에서 실행하지 않았다.

두 40-epoch 학습이 완료되면 `tools/evaluate_v1_sampling_pair.py`가 같은 전체21 validation case를 평가한다. case별 MRR/top1/margin을 집계하고 case 단위 paired 차이를 보고한다. 평가 자원 예산과 bootstrap 반복 수·confidence는 필수 입력이며 코드가 임의의 품질 허용 오차를 만들지 않는다.

먼저 판별할 질문은 **작게 만든 그래프에서도 v1의 ranking 학습 및 validation 품질이 유지되는가**이다. 이후에 전체128 candidate inference/transform과 Basic CP80 대비 nnU-Net 효용을 검증한다. 1회 cold update의 약8.46배 compute 감소를 whole-epoch 배속이나 품질 유지로 환산하지 않는다. loader·sampling·GPU update·validation·저장을 포함한 실제 epoch 비용은 장기 비교에서 기록해야 한다.

## 상태 구분

- 구현 및 기존 정적/단위 검사: 기존 receipt로 확인됨.
- 실제 CT/CUDA mechanical smoke: 기존 r4 runtime416 evidence로 확인됨.
- ZIP standalone 재현성: 로컬 및 전달된 독립 검토의 28개 검사로 확인됨.
- 전체84/21 학습·전체21 평가·전체128 inference·nnU-Net 비교: 미실행.
- ranking 품질 보존·최적 graph 크기·전체 epoch 가속: 아직 미검증.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 기존 승인된 대조를 고정했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 공통 calibration lock과 batched 경로를 확인했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 실제 GPU evidence의 범위를 유지했으며 새로운 실행 자원 측정은 하지 않았다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 문서 작업에는 새 OOM이나 규모 변경이 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 실제 CUDA smoke를 유지했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 이번에는 검토 마감 기록만 추가했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
