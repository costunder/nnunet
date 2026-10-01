# v2.2 — 같은 BN에서 실제 Adam 이동 후 순위 점수 비교

## 확인된 문제와 이번 수정

서버의 `reference_causal_20261001_230955.json` 재출력에서 전체 후보 점수가 크게
함께 이동하는 동안 후보 간 표준편차가 약1e−6에 머물렀다. 기존 reference
tile-before/after에는 native train forward가 갱신한 BN running statistics도
섞여 있었다. 이 때문에 기존 parameter cosine과 평가 전후만으로 CE/alignment의
효과를 확정할 수 없었다. 상세 원문·수치는 기존 서버 결과 문서에 보존했다.

`--finite-shadow-score`는 이 통제 한계를 해결하는 **진단 전용** 선택이다.
Production 모델·loss·훈련 설정을 바꾸지 않는다. `--causal-probe`와 원래 full
physical P/U tile을 고르는 `rankable_full_batch_prefix`가 필요하다.

원래 train forward **전**의 buffer snapshot을 저장한다. 원래 forward에서
weighted rank/CE/alignment/full gradient를 계산한 뒤, 현재 clone AdamW moments와
원래 global clipping·weight decay를 복제한 기존 shadow를 사용한다.

| 평가 arm | 적용하는 parameter | 공통 조건 |
| --- | --- | --- |
| no_change | update 이전 parameter | 같은 native CT·support·teacher·pre-forward BN·dropout-off |
| rank_only | weighted rank gradient로 복제 AdamW가 만든 정확한 post-parameter | 위와 동일 |
| full | 원래 전체 objective로 복제 AdamW가 만든 정확한 post-parameter | 위와 동일 |

각 arm에서 **CNN부터 다시 실행**한다. 옛 L0 embedding을 재사용하지 않는다.
FP32 `old + (new-old)` 재조합 대신 shadow의 정확한 post-parameter를 복사하고
hash를 검사한다. Teacher 재학습·refit은 하지 않는다. Query target은 기존 loss와
평가에만 쓰고 모델 입력을 바꾸지 않는다. 세 arm의 입력·buffer·teacher·RNG hash와
원래 model·gradient·mode·optimizer 보존을 검사한다. 원래 full update의 exact
shadow parity도 유지한다. 입력 결속과 자원 검사 실패를 우회하지 않는다.

원래 train-mode gradient를 deterministic eval-mode로 읽는 비교이므로 두 함수는
다르다. 또한 eval 모드 자체가 GPU kernel 결정론을 보장하지는 않는다. 실제 backend
설정은 보고서에 기록하고 helper가 바꾸지 않는다. 작은 점수 차이를 성능 개선으로
확대하지 않는다. Train-only linear head fit과 held-out 평가는 이번 범위가 아니다.

## 비용과 출력

매 tile에 no-change/rank-only/full의 native inference3회가 추가된다. 모듈 채널·
L1/L2 층수·loss·Basic CP·전체 observation·후보128·physical batch·원본 mask는
그대로다. 추가 clone·hash·readout 비용은 별도로 기록하고 update timing에서
callback 비용을 분리한다. 이 진단 시간은 production epoch 시간으로 환산하지 않는다.

JSON에는 각 arm의 margin·score std/range·동점률·pair loss와 no-change 대비
공통 점수 평균 이동 및 P−U 차이 이동을 저장한다. Console에도 한 번에 표시한다.
이전 JSON에는 arm을 만들어 넣지 않고 기존 출력 그대로 읽는다. 실패한 arm을
측정 완료로 표시하지 않는다. Production checkpoint·ready 표시를 만들지 않는다.

## 로컬 검증 — 성능 평가와 구분

RTX5070Ti 실제 CT `liver_66`, 같은 donor, 전체 관측133=양성5+미관측128,
원래 DEBUG context139와 저장 support6을 사용했다. Physical/effective batch32,
gradient accumulation1, FP32, CNN channels12/24/32, hidden128, 4heads,
L1두 층·L2두 층을 유지했다. 비교 branch별2회, 총4회 update와12회 finite
inference를 검사했다. Full shadow/actual delta는 모두 exact hash로 일치했다.

모든 arm의 buffer·native input·teacher hash가 같고 각 arm CNN을 재인코딩했다.
원래 checkpoint·support·assignment·model·RNG는 보존됐다. 결과는
`validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json`이다.
Load 이후 elapsed14.22초, peak allocated.780GiB, peak reserved.936GiB,
RSS2.30GiB, workers4, explicit CUDA/RSS/resident budget3/12/8GiB였다.
GPU1개·16 CPU logical core·약43.4GiB available RAM을 확인했다.
Legacy 초기1,125,718 parameter; reference 전이 후 구성은 report에 별도로 기록된다.

CUDA/admission29개(새 수치 검사4개 포함), 새 출력3개, 기존 출력/CLI42개 검사를
통과했다. 수치 검사는 CUDA, 문자열/JSON 검사는 표준 라이브러리를 사용했다.
UNIT의 작은 합성 입력과 실제 CT 검사를 구분한다. UNIT에만 적용한 TF32-off·
kernel 결정론 설정은 복원했다. 실제 CT와 helper는 backend 정책을 바꾸지 않았다.

이 검사는 로컬 DEBUG 가중치다. 서버의 오래 학습한 가중치에서 rank-only와
full 중 무엇이 개선되는지는 **아직 미실행**이다. 이번 로컬 arm의 점수 상승을
서버 붕괴 해결이나 CP 효과로 보고하지 않는다. 장기 GNN·nnU-Net 학습은 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 명시적 DEBUG만 사용했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 원래32 유지.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 UNIT는 별도 표시.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
