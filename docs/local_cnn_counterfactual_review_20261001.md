# v2.2 학습 부진 후속 검토와 원인 분리 진단

사용자의 상세 A~E 검토(`9cecae3f-61c6-4a2c-9c1e-301c34ef2380` 첨부)를
끝까지 읽고 실제 구현과 대조했다. 서버 증거는 epoch 필드22 / step11872의
텍스트 요약이며, 원본 서버 JSON과 학습 가중치는 로컬에 없다. 아래 새 probe의
로컬 결과는 해당 서버 모델의 측정으로 표시하지 않는다.

## 근거와 판단

현재 CNN이 상수 출력을 만든다는 설명은 서버 자료와 맞지 않는다. 후보 차이는
CNN map과 recipient mean/project에 남아 있다. 가장 큰 수축은 fusion과 두 번째
query L1이다. 39~120배는 **fusion input→output**의 정규화 분산 비율이다.
recipient project→fusion output 비율은 별개이며 약125~381배다. 이 비율은
정확도나 종양 정보 손실률이 아니다.

Residual은 이미 존재한다. FFN·공통 message·norm을 분리하지 않고 residual을
더 넣거나 CNN을 키우는 패치를 하지 않았다. Message scale0에서도 pair win은
거의 변하지 않았으므로 분산 증가만으로 해결됐다고 판단하지 않는다.

현재 masked mean은 후보별 **국소 crop 안의 간 영역**에 적용된다. 간 전체
CT를 한 번 평균낸 모델이라는 설명과는 다르다. 다만 crop 안의 공간적 배치가
평균으로 줄어드는 문제는 남으며, v1의 role/shell별 pooling과 같은 구조는 아니다.

v1은 다른 후보/양성 정의와 loss·scoring path를 사용했다. v1의 높은 순위 결과는
학습 가능성을 보여 주는 비교 근거지만 현재128개 미관측 위치의 추천 정확도와
동일한 평가가 아니다. 현재 softplus는 margin0에서도 derivative가-0.5이므로
loss가 ln2 근처라는 이유만으로 loss 자체에 학습 압력이 없다고 판단하지 않는다.

## 추가한 진단과 production 경계

| 진단 | 실제 비교 | 보존 조건 |
|---|---|---|
| query FFN | L1 **2층의 FF residual만** 0/.25/.5/1, message1 | support FFN·L2·층수·physical batch 유지; scale1 production parity |
| attention | head별 후보/source weight 분산, 모든 후보 쌍 cosine/JS, entropy | source·후보 생략 없음; 통계만 FP64, 원래 attention은 FP32 |
| fusion bypass | 기존 Fuse baseline과 `r + λ·Fuse` | query와 **전체 유효 train support**를 동일 수식으로 계산; λ마다 plan 새로 fit |
| Adam history | saved moments에서 모든 trainable parameter에 **실제 zero gradient**를 넣은 다섯 번째 복제 step | saved weight decay·clip·layout·moment·RNG 보존; 원본 update0 |
| alignment audit | 실제 epoch의 case/group별 P/U/tile 수와 `K_g/S` 가중치 | 전체 P×U/관측 coverage와 CE 정규화 검사; loss 변경 없음 |
| direct scalar head | fusion 이전 recipient project→별도 MLP→scalar | CNN·원본 scoring path 고정; 새로운 head만 명시한 짧은 진단 step으로 학습 |

Fusion에서 λ1은 기존 모델이 아니다. 기존 Fuse만 사용하는 branch를 별도 baseline으로
둔다. Saved epoch memory를 새 query와 섞지 않으며, baseline도 현재 가중치로 support를
재인코딩한다. 따라서 기존 보고서의 오래된 memory baseline과도 구분한다.
모든 λ는 한 번 구한 recipient/Fuse 128D 벡터를 공유한다. 선택된 train query의
trace 결과도 재사용하며, support의 나머지 원본 record만 한 번 추가 인코딩한다.
3D map·CT를 새 영구 캐시로 만들지 않는다. 전체 support 재인코딩 비용은 별도 기록한다.

Saved memory의 record 순서·owner·class·**donor group**도 inventory와 일치시킨다.
Query group은 support의 recipient/donor 양쪽에서 제외한다. 모든 후보·간 mask·
native CT·범위·입력 계약 검사를 유지한다.

History delta는 과거 moment뿐 아니라 weight decay와 Adam의 분모도 포함한다.
`full-history`, `rank-history` 차분은 비선형 counterfactual이며 선형 기여율로
해석하지 않는다. Saved exp_avg와 `(1-beta1)*clipped current gradient`도 따로 기록한다.

Direct head는 보존된 v1 scalar head의 `4h→2h→1`, LayerNorm/SiLU/dropout.1 계열을
사용하되 입력은 현재 recipient128D다. v1 전체 계층·가중치·loss를 복제한 모델이
아니다. 새 optimizer의 rank-only 진단과 과거 full-objective Adam을 쓴 원래 모델의
차이도 존재한다. 훈련 fit만으로 CNN의 CP 특징 유효성이나 원래 실패의 단일 원인을
입증하지 않으며, 짧은 probe 실패도 CNN이 쓸모없다는 증거가 아니다. 학습/검증 case와
record는 분리하고 동점 순서는 원래 geometry hash 규칙을 사용한다.

## Alignment 실제 metadata 감사

로컬 보존 전체 paired inventory에 현재 donor 배정을 재구성해 비교했다.
원본 inventory SHA256:
`4e711671eaf0c92859230a71ac6bd6a716a68bf7ed439337b3ed27deaa9561c3`.
서버 native inventory를 내려받은 독립 검증은 아니며, 서버 실행에서는 해당
checkpoint에 결속된 Dataset으로 동일 감사를 다시 수행한다.

- Batch32·epoch22: train11,279관측, 84group, 533tile.
- 전체 P×U67,456쌍: 누락/중복0. CE 두 class의 step-mean coefficient mass는 각각0.5.
- 그룹별 K_g=4~38. 현재 alignment의 최대/최소 coefficient mass 차이는9.5배.
- liver_117: P71/U128, K38, 전체 step-mean weight7.129%; 균등 group1.190%의 약5.99배.
- `S/(G*K_g)`는 **미적용** 비교계수다. 현재 alignment는 optimizer-step weighted다.

모델·dropout은 tile 사이에 변하므로 매번 동일 scalar loss를 복제했다는 주장은 하지
않는다. 확인한 것은 coefficient mass다. Group-balanced와 step-weighted 중 어느
목적함수를 사용할지 명시하기 전 production loss를 자동 변경하지 않는다.

## 검증과 남은 범위

기존 검사와 새 FFN/attention/history/alignment/fusion/direct-head/요약 검사
총68개가 통과했다.
CUDA 검사는 실제 RTX5070Ti에서 실행됐고 skip하지 않았다. 실제 CT의 기존
DEBUG checkpoint(epoch0/step1, saved batch2, 전체 모델1,125,718parameter)로
네 가지 counterfactual 전체 경로도 통과했다. 전체 eligible support union8개에서
이미 trace한2개를 재사용하고6개만 추가 인코딩했다. 해당 support L0 basis 준비
약26~27초, peakCUDA약0.235GiB. 원본 weight/payload/support/plan/RNG 보존을 확인했다.

이 DEBUG는 입력/연산/결속/보고서 검증이다. 서버22epoch 모델의 원인 판정이나
추천 성능 개선이 아니다. Direct-head를 포함한 두 번째 실제 CT DEBUG 통합 검사도
통과했다. 새 head만8회 update했으며 CNN update0, 원본 feature/RNG 보존을
확인했다. 훈련 pair loss는 낮아졌지만 검증 순위는 개선되지 않았다. 소규모
기계적 검사에서 성능 개선을 주장하지 않는다. 최신 report와 receipt는
`work/local_cnn_counterfactual_direct_DEBUG_20261001/report.json`,
`validation/local_cnn_counterfactual_20261001/debug_receipt.json`이다.
Production architecture/loss·Basic CP·L1/L2 크기·
split/seed42·128후보·전체 관측·원본 paste mask는 변경하지 않았다.

추가 서버 진단 결과에서 순위 변화까지 확인한 다음 fusion→query FFN→attention
순서로 반응하는 지점을 패치한다. 구조/loss 변경은 새 계약으로 기록하며 현재
checkpoint의 exact resume 패치라고 숨기지 않는다. 장기 GNN/nnU-Net 학습은
자동 시작하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 진단 case/step 범위는 명시한다.
- [x] Saved physical batch와 병렬 loader를 보존하고 support CNN 결과를 재사용했다.
- [x] GPU와 CPU/RAM 예산을 확인하며 실제 CUDA 검사와 peak/RSS를 기록한다.
- [x] 메모리·비용 문제를 모델 축소나 sample skip으로 숨기지 않았다.
- [x] DEBUG와 production을 분리했다.
- [x] 합성 unit 입력과 실제 CT DEBUG를 구분하고 fallback을 쓰지 않았다.
- [x] Production parity와 복제 loss/gradient/optimizer 경로를 검사했다.
- [x] 실제 설정과 counterfactual 경계를 명시했다.
- [x] 단위/smoke와 전체 학습·평가 및 서버 원인 판정을 구분했다.
