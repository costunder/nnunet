# V1/A/B/C 공통 전체후보 서버 결과와 다음 진단

## 사용자 제공 서버 결과

사용자가 `ece-a6gpu6`에서 `16f7eb8`의 공통 평가를 실행한 터미널 결과를
제공했다. 원시 서버 JSON을 로컬에서 읽은 것은 아니다. 보존 기록은
`validation/v17_historical_full128_server_20261005/summary_from_user.json`이다.

실제 결과 경로는
`/home/aicompetition06/Medical/experiments/v17_V1_A_B_C_full128_20261005_212328/summary.json`이다.
새 학습을 시작한 결과가 아니라 각 기존 own-task BEST의 평가 결과다.

| Arm | case-first-P MRR | Hit@1 | P/U pair-win | softplus loss |
| --- | ---: | ---: | ---: | ---: |
| V1 | 0.233839 | 0.125000 | 0.596528 | 1.935892 |
| A | 0.146013 | 0.062500 | 0.621470 | 3.393549 |
| B | 0.226487 | 0.125000 | 0.561574 | 1.234715 |
| C | 0.195354 | 0.125000 | 0.562500 | 1.131561 |

V1-B의 MRR 차이는 0.007352다. 이 집계만으로 통계적 우열을 정하지 않는다.
Hit@1의 분모는 순위 평가 가능한 환자 수이며, P가 없는 환자는 전부 score한
뒤 순위 분모에서 제외한다. 21명을 분모로 환자 성공 수를 역산하지 않는다.

## 확인된 사실과 해석 제한

V1도 전체 관측 P+128U에서 이전 원래8후보 MRR1.0을 유지하지 못했다.
따라서 CNN 또는 native upper로 교체한 부품만으로 전체 실패를 설명하는
가설은 충분하지 않다. V1은 원래 task를 학습했다는 사실과 새 task에서 높은
순위를 유지하지 못했다는 사실을 함께 보존한다.

A는 pair-win이 가장 높지만 MRR/Hit@1이 가장 낮다. 많은 P/U 비교의 평균
분리와 가장 높은 U를 넘는 순위는 다르다. C의 loss가 가장 낮다고 C가 추천을
가장 잘한다고 판정하지 않는다. scorer와 점수 scale도 arm 사이에 다르다.

이번 실험은 후보 개수만8→128로 바꾼 대비가 아니다. 원래 source-anchor와
corrupted curriculum 비교 과제에서 관측 P/U 과제로 바뀌었고, own-source에서
독립 donor 조건으로 바뀌었으며 sample 지표와 case-first-P 지표도 달라진다.
V1/A는 recipient lesion annotation을 사용하는 외부 donor adapter이고,
B/C는 원래 anchor/curriculum support를 유지한다. 이 경계를 숨기지 않는다.

현재 GT는 그대로다. P=실제 관측된 적격 종양 위치, U=미관측 비교 위치다.
P를 특정 donor의 CP 적합 정답으로, U를 CP 부적합 정답으로 바꾸지 않는다.
blind CP recommendation quality나 nnU-Net segmentation 결과로 승격하지 않는다.

## 재학습 없는 경쟁자 개수 진단

`tools/analyze_full128_competition.py --summary <위 summary.json>`은 저장된
arm별 `report.json`의 실제 점수와 GT-independent 정렬 순서를 읽는다.
summary/완료 receipt/report SHA/동일 cohort와 전체 record coverage를 검사한다.
원본 입력, 모델, checkpoint, report를 수정하지 않는다. CT decode나 neural
forward 없이 정확한 조합 확률을 계산한다.

각 환자의 모든 P를 유지하고 U를 k=7/15/31/63/128개 균등하게 고르는 분석적
부분집합에 대해 기대 MRR과 Hit@1을 계산한다. 원래 curriculum8을 재현하는
것이 아니므로 결과 이름도 P+7U이며, 실제 총 후보 수는 P 수에 따라 다르다.
random sampling을 실행하지 않고 모든 가능한 U 부분집합의 기대값을 구한다.

저장된 전체 순서에서 첫 P 앞에 놓인 U 수를 K라 하면, 부분집합의 첫 P rank는
`1 + H`, `H ~ Hypergeometric(128, K, k)`다. 따라서

`E[MRR] = Σ_h Pr(H=h)/(h+1)`

`E[Hit@1] = Pr(H=0)`

이다. k128 endpoint는 원래 full-case MRR/Hit@1과 일치해야 한다. P가 없는
환자는 계속 coverage에 포함하며 순위 기대값을 NOT_EVALUABLE로 남긴다.

이는 **full-case에서 이미 계산된 score를 고정한 경쟁자 수 효과**만 분리한다.
V1/A의 joint upper를 작은 후보 그래프로 다시 실행한 결과와 같다고 주장하지
않는다. donor나 GT가 바뀐 효과, 학습에서128U를 보았을 효과도 측정하지 않는다.

P+7U에서도 낮으면 경쟁자 수 증가만으로 설명하기 어렵다. P+7U가 높고128U가
낮으면 경쟁자 증가의 영향은 확인되지만, 원래 source-anchor curriculum과
현재 관측 과제 사이의 차이가 해결된 것은 아니다. 별도 새 장기 학습을 자동
시작하지 않는다. D는 자신의 기존 상태와 계약을 유지하며 이번 진단이 건드리지 않는다.

## 검증 상태

UNIT 검사 **18개 PASS**다. 정확한 조합 확률은 population 0~7의 모든
ahead/draws 및 가능한 부분집합을 exhaustive enumeration으로 대조했다.
입력 변조·누락·중복·비유한 score·잘못된 정렬·분모와 endpoint 불일치,
완료 seal/원래 request/summary 불일치를 명시적으로 거부한다.
기존 실제 CT/CUDA full128 DEBUG report 4arm에 대한 JSON 분석도 통과했다.
이는 저장된 JSON의 분석이며 새 CPU/GPU neural run이 아니다. 각 arm의 k128
endpoint는 기존 MRR/Hit@1과 일치했고 원본 JSON bytes를 보존했다.
결과는 `validation/v17_full128_competition_DEBUG_20261005.json`이다.
그 입력의 성능 숫자를 production 품질 근거로 사용하지 않는다.

서버의 새 경쟁자 curve는 사용자가 위 summary로 실행한 뒤에만 알 수 있다.
이번 서버 summary에 없는 값, paired CI와 정확한 환자 성공 수는 만들지 않는다.

## 예전 원본 V1 30mm checkpoint를 추가할 때

사용자가 기존 서버의 30mm V1을 같은 평가에 추가하자고 요청했다. 실제 보존
로그에서 `HierCP/work/full/model.pt`와
`HierCP/work/paired_basic_vs_hiercp/folds/fold_0/gnn/model.pt`를 확인했다.
로그 경로를 찾은 사실과 현재 서버 파일의 존재·상태를 확인한 사실은 구분한다.
`v1_native_nested416_seed42_20261003/native`는 최근 준비 실패가 기록된 별도
실험이므로 오래전 완료된 V1로 자동 선택하지 않는다.

원본 설정은 `adaptive_roi_margin_mm=30` / `context_outer_radius_mm=28`이다.
bounded30은 outer30과 runtime scope patch를 사용하므로 원본 native30과
같다고 표시하지 않는다. 현재 10mm 평가기의 경로만 바꿔서 실행하지 않는다.

과거 full 실험의 105/26 split과 현재 84/21 split이 다르므로 저장된 실제
학습·prototype case IDs와 현재 평가 21case의 교집합부터 확인해야 한다.
훈련에 사용한 case를 held-out이라고 표시하거나 비교 cohort를 몰래 줄이지
않는다. 기존 checkpoint의 구조·범위·BEST·source 계약을 확인한 뒤에만
native 전용 GPU 평가기에 연결한다. 초기 `3a50628`의
`tools/inspect_native_v1_checkpoint.py`는 기록된 위 두 경로와 native suite
경로를 모두 조회하도록 작성됐으며 checkpoint의 실제
구조·범위·학습 완료 필드·training/prototype 환자 겹침을 출력한다.
폴더 이름이나 현재 옆에 있는 config를 학습 설정의 증거로 삼지 않는다.
명시적 저장 train case IDs 또는 checkpoint SHA로 결속된 cache index의
실제 case_id가 없으면 학습 환자 목록은 UNKNOWN으로 남긴다.
조회기 UNIT **11개 PASS**이며 새 metadata 보고서는 원본 실험 디렉터리 밖에
exclusive create로 저장한다. 18개 경쟁자 분석 검사와 합계 **29개 PASS**다.
`3a50628`에서는 원본 checkpoint와 평가 대상 metadata를 읽는 서버 조회만 준비했으며,
30mm GPU 평가나 새로운 40epoch 학습을 시작하지 않았다.

## 원본 30mm 조회의 실제 서버 결과와 출력 정정 — 2026-10-06

사용자가 `3a50628` 조회의 서버 출력을 제공했다. 이 절은 그 출력에 대한
기록이며, 로컬에서 서버 원시 JSON이나 서버 checkpoint를 읽은 결과가 아니다.
원시 조회 보고서는
`/home/aicompetition06/Medical/experiments/v1_native30_inspection_20261005_235901.json`이다.

| 실제 checkpoint | 서버 조회 | 저장된 선택 epoch | 완료 표시 | 저장된 범위 |
| --- | --- | ---: | --- | --- |
| `/home/aicompetition06/Medical/HierCP/work/full/model.pt` | READ_CPU_METADATA | 30 | true | native ROI30 / context28 |
| `/home/aicompetition06/Medical/HierCP/work/paired_basic_vs_hiercp/folds/fold_0/gnn/model.pt` | READ_CPU_METADATA | 24 | true | native ROI30 / context28 |

**예전 원본 30mm checkpoint 두 개는 존재하고 실제로 읽혔다.** 별도의
`v1_native_nested416_seed42_20261003/native/results/v1.0/checkpoint_best.pt`만
MISSING이었다. 이는 이전 준비 실패가 기록된 최근 native suite 경로다.
오래전 완료한 30mm 실험의 checkpoint와 같은 파일이 아니다. 그 별도 경로를
기본 후보에 함께 넣은 출력이 사용자의 평가 대상과 맞지 않았다. 기본 조회를
위 두 역사적 모델로 한정하고, 최근 suite는 명시적 옵션으로만 조회한다.

`architecture=UNKNOWN`은 가중치가 없다는 의미가 아니다. 보존된 이전 원본
pipeline의 `static_checkpoint_metadata`는 `method`, `framework`, `model_kwargs`,
`graph_config`, `ct_clip`, `prototype_training_cases`, `prototype_fingerprint`를
저장하며 `architecture_version`은 저장하지 않았다. 실제 구조 인자는 저장된
`model_kwargs`에서 읽을 수 있다. 이것과 실제 모델 strict load 및 연산 검증은
다른 확인이므로 구조 tag의 부재를 실행 실패로 표시하지도, 모델 검증 완료로
바꾸지도 않는다. `preflight` sidecar의 부재 역시 읽은 checkpoint 파일의
부재가 아니다.

이전 검사기의 training ID 복원은 새로운 `cache_publication.index_sha256`
필드를 요구했지만 옛 저장 형식에는 그 필드가 없다. 옛 checkpoint의 실제
`training_signature.train_cache_files`와 현재 cache index의 명시적
`entries.case_id`/`split`을 대조하는 정보는 별도 재구성 evidence로 남긴다.
이는 파일명에서 환자 ID를 추측하는 방법이 아니다. 단, 학습 당시 index bytes의
SHA 결속은 없으므로 명시적으로 저장된 training case IDs 또는 SHA 결속된
evidence와 동일한 강도의 독립성 확인으로 승격하지 않는다.

서버 출력에서 full 모델의 prototype 환자와 현재 21case의 교집합은 17명,
paired fold0 모델의 prototype 교집합은 0명이었다. 이 숫자는 prototype
교집합이며 optimizer 학습 환자 수를 뜻하지 않는다. paired fold0가 우선
대조 대상이지만 현재 21case와 실제 학습 split의 대응 및 원본 연산 경로는
별도로 확인해야 한다. 현재 10mm 전용 GPU 평가기에 경로만 바꿔 넣지 않는다.

이번 조회로 원본 30mm의 128후보 점수, MRR, Hit@1은 아직 측정되지 않았다.
가중치 재생성, 재학습, 기존 checkpoint 수정·삭제는 수행하지 않는다.
조회 출력 수정의 로컬 UNIT 검사 **16/16 PASS**다. 기본 두 파일·최근 suite
명시적 선택·구형 tag 부재·비어 있는 state dict·부가 JSON 부재·보조 mapping과
held-out 판정의 분리·입력 변조 및 원본 보존을 검사했다. UNIT fixture 검사이며
서버 checkpoint 재조회나 CUDA 모델 평가를 수행한 결과가 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 분석은 저장된 전체 score를 읽으며 모든 P와 환자를 보존한다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. k는 명시된 분석 변수이며 neural 실행 설정이 아니다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 새 모델 실행 없이 작은 조합 기대값을 계산한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 새 GPU 실행을 하지 않고 기존 실제 CUDA evidence를 사용한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 JSON 진단에는 모델 allocation이 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 진단이 실제 저장 score→고정 순위→정확한 기대 metric에 연결된다. 새 loss/gradient/optimizer를 만들지 않는다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
