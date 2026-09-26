# v2.2 — 관측 종양 순위 학습과 추천 시 제외

사용자 합의: 실제 관측 종양을 순위 학습에 포함하고 상위로 올리도록 학습한다. 추천 시에는 관측 종양과 겹치는 위치를 제외하고 남은 유효 후보에서 선택한다. 정답 마스크로 점수를 올리거나 무조건 1위만 버리지 않는다.

이 변경의 objective ID는 `observed_rank_v1`이다. v2.2 계열의 paired v1 L0 + 기존 v2.22 L1/L2 위에 적용하며, 기존 분류 학습은 `observation_ce`로 보존한다. 설정은 `config/v22_observed_ranking.json`, 새 trainer는 `tools/v22_ranking_training.py`다. frozen 원본 config/model/cache 소스를 수정하지 않아 원본 graph cache는 그대로 사용한다.

## 실제 학습 목표

모델은 donor/recipient CT 문맥과 query 그룹을 양쪽에서 제외한 inner-train support로 점수를 계산한다. query의 target은 그 뒤 loss 계산에만 사용한다. 점수는 두 관측 class logit의 차이 `s = logit_observed - logit_unobserved`이며 softmax 확률과 순서는 같지만 교정된 CP 성공 확률은 아니다.

같은 CT에서 관측된 양성 위치 `p`와 미관측 비교 위치 `u`에 대해 다음 순위 손실을 계산한다.

`rank_loss = mean(softplus(-(s[p] - s[u])))`

`total_loss = rank_loss + observed_presence_CE + 기존 L2 alignment_loss`

관측 분류 CE는 기존 종양 존재 관측을 맞히는 보조 목적이다. 미관측 위치를 **붙이면 안 되는 위치라는 정답**으로 재정의하지 않는다. 다만 관측 위치를 미관측 위치보다 우선한다는 약한 순위 가정은 존재한다. 두 mean logistic 목적에 단위 가중치를 주었으며, 기존 L2 weight1을 유지했다. 이 가중치가 최적이라는 근거는 없고 비교 실험이 필요하다. BPR의 상대 순위 원리를 참고한 프로젝트 변형이며 원 논문의 완전 재현은 아니다. [BPR](https://arxiv.org/abs/1205.2618).

현재 캐시에는 inner-train84케이스/11,279관측 중 eligible small-tumor anchor527개가 있고, 19케이스에는 그 양성이 없다. inner-val21케이스/2,823관측 중 양성135개, 무양성5케이스다. 기존 크기 기준과 모든 관측을 보존한다. 무양성 케이스를 버리거나 정답을 만들지 않고 관측 보조목적/L2 학습을 유지하며, 해당 케이스의 종양 순위 평가는 불가능하다고 별도 집계한다.

학습 positive는 보존된 캐시의 **적격 소형 종양 anchor 전체**다. 모든 크기의 종양을 새 positive로 추가한 변경은 아니다. 반면 추천 제외에는 **크기와 무관한 모든 주석 종양**을 사용한다.

## 전체 비교 범위와 계산 방식

- 현재 physical batch의 각 query와 같은 CT에 속한 **전체 관측 후보**를 비교 대상으로 잡는다. 같은 batch에 양성이 없어도 epoch support 생성 때 계산한 그 CT의 L0 임베딩을 참조한다.
- 참조 L0 임베딩은 detached epoch memory이고, 현재 batch 위치는 현재 L0 forward 값으로 교체한다. 같은 case의 P/U 쌍 중 현재 batch가 한쪽 이상에 참여하는 모든 쌍을 계산한다. 노드·엣지·후보 cap이나 query 생략은 없다.
- L1/L2 support 준비는 한 번 하고 현재 batch와 참조 후보의 점수를 함께 계산한다. 참조 후보의 관측 target은 이 scoring 경로에 들어가지 않는다. 자기 환자의 참조 임베딩을 loss 비교에 사용하는 것과 자기 GT를 L1 support로 노출하는 것을 구분한다.
- 현재 query L0와 L1/L2는 gradient를 받는다. 참조 후보의 L0는 epoch 시점의 값이므로 **전체 CT 후보를 동시에 최신 L0로 역전파하는 정확한 full-case 목적은 아니다.** 같은 pair가 서로 다른 batch에서 비교될 수 있다. 이는 메모리 기반 minibatch 순위 학습이며, 추가적인 품질/refresh ablation 대상이다.
- 기존 모델5,550,806 parameters, L0/L1/L2 깊이3/2/2, hidden128, heads4, CNN12/24/32, patch48³, seed42, GNN40epochs, CP80%, 전체 split/관측, accumulation1을 유지한다. Physical batch와 worker는 실제 새 loss calibration으로 결정하고 원래 saved batch를 축소하지 않는다.

## 평가와 best checkpoint

inner-validation의 모든 관측 위치를 **제외하기 전** 점수로 평가한다. 정답 위치를 인위적으로 먼저 정렬하지 않는다. 동점은 관측 종양에 불리한 순위로 계산해 GT를 이용한 동점 이득을 막는다.

- `ranking_pairwise_loss`: 같은 CT 관측 종양/미관측 후보 상대 순위 손실. 이것으로 best epoch를 선택한다.
- `ranking_recall_at_1/5/10`: 전체 eligible 관측 종양 중 해당 순위 안에 든 비율.
- `ranking_mrr`: 양성이 있는 CT별 첫 관측 종양 순위의 역수 평균.
- `ranking_mean_observed_rank`, 평가 가능한 CT 수, 양성 없는 CT 수.
- 관측 CE/accuracy/AUROC는 별도 보조 지표로 유지한다. 순위나 분할 Dice를 대신하지 않는다.

`metrics.csv`, epoch JSON, `ranking_epoch_XXX.json`에 저장한다. 모든 종양의 상위 진입은 학습 목표이며 강제 보장하지 않는다. 여러 종양이 있으면 Recall@1은 구조상 모두를 포괄할 수 없으므로 k별 recall과 개별 rank를 함께 확인해야 한다.

## 추천 경로

`tools/v22_rank_recommendation.py`의 `load_checkpoint()`는 새 objective·stride4·core source identity·train-only support를 확인하고 모델을 로드한다. DEBUG 가중치는 기본적으로 거부한다.

`recommend()`는 동일 recipient와 동일 donor component의 모든 제공 후보를 먼저 scoring한다. 이후 `rank_then_filter()`가 실제 recipient 간/종양 label과 **recipient spacing/변환이 적용된 전체 paste mask 및 동일 anchor**로 제외를 계산한다. 중심점만 검사하지 않고, 임의의 종양 거리 제한이나 마스크 팽창을 추가하지 않는다.

1. model score만으로 원래 순위를 계산한다.
2. 실제 paste footprint가 어느 주석 종양과든 겹치면 제외한다.
3. 영상 경계와 기존 최소 간 포함률 조건도 확인한다.
4. 남은 후보 중 최고 점수를 선택한다. 모두 제외되면 `selected_index=None`, `keep_original=True`를 반환한다.

원래 점수·순위, 중심 종양 관측 여부, 겹치는 voxel 수, 제외 사유, 제외 후 순위를 반환한다. 고정 donor의 후보 비교가 필요하며 서로 다른 donor의 records를 한 CP 이벤트로 넘기면 오류를 낸다. 실제 paste와 다른 좌표계/anchor를 전달해서는 안 된다.

**현재 paired online CP bank/nnU-Net trainer에서 이 API를 실제 CP 이벤트에 호출하는 통합은 아직 미완료다.** 이번 완료 범위는 순위 학습/평가·checkpoint·paired 후보 scoring/제외 API 및 그 검증이다. 기존 legacy `v1_local.score_candidates()` 호출이 자동으로 새 API로 바뀌었다고 가정하면 안 된다.

## 검증 결과와 한계

- 관련58회귀 검사 통과. 수학적 rank gradient 방향, batch에 양성 없는 경우, CT 간 비교 차단, 동점, 전체 footprint 겹침, 모든 종양 제외, 무후보 원본 유지, objective 재개 거부를 검사했다.
- 실제 CT batch32, 모든 graph node/edge 유지:324,992 nodes/21,995,327 edges. liver_108 전체187관측+다른3case6support =193개 reference를 사용했다. live query32개가 모두 미관측이어도59양성×32=1,888 ranking pair를 계산했다.
- **rank loss만으로 L0에 유한한 nonzero gradient**를 확인했다. Joint loss의 모든 학습 파라미터 gradient도 존재하고 유한했다. Peak allocated6,184,051,712 bytes, reserved6,459,228,160 bytes. 추가 rank-only gradient probe 때문에 이 step timing을 최종 throughput으로 해석하지 않는다.
- 같은 저장 상태의 연속 실행과 재개 실행에서 loss/model/Adam bitwise 일치. 전체 support는 사용하지 않은 명시적 DEBUG다.
- 별도 actual CT train8/val2/1epoch/physical2 DEBUG에서 step1 저장중단→재개→4전체update→support refresh→순위평가→best checkpoint→최종 support 완료. DEBUG 전용 설정이며 production40epochs/전체데이터와 분리했다. 이 테스트의 Recall@1=0도 그대로 보존했다. 실행 성공을 순위 성능 성공으로 바꾸지 않았다.
- 실제 동일 donor(liver_101 component15), recipient liver_108의2후보와 원본 label 검사: 1위 실제 종양 위치3,557voxel overlap 제외; 2위 중심에는 종양이 없지만 전체 donor footprint가497voxel 겹쳐 제외; 결과는 원본 유지였다. 두 후보만 사용한 API DEBUG이며128후보 CP 성공률 검사가 아니다.
- 최종 실제 DEBUG artifact load 성공, production loader의 DEBUG 거부 확인. 증거는 `validation/v222_r6/observed_ranking_20260927_DEBUG.json`에 모았다.

관측 종양 본체를 보는 shortcut, donor가 달라져도 추천이 같은 문제, 같은 사람 재검사 여부, rank1 종양을 제외한 뒤의 CP 효용은 아직 검증되지 않았다. 기존 paired cache는 위치마다 donor가 다를 수 있어 동일 donor 조건부 순위 효과를 보장하지 않는다. donor-swap/recipient-only/CT 중심 통제 및 같은 nnU-Net 조건에서 CP 비교가 필요하다. 입력을 임의로 가리거나 label을 다시 만들지는 않았다.

## 재개 및 실행

새 process run 기본은 `observed_rank_v1` + stride4다. 서버 runner 기본 runtime도 process로 변경했다. `--training-objective observed_rank_v1`로 명시할 수 있다. 기존 checkpoint는 objective 필드가 없으면 `observation_ce`를 유지한다. 서로 다른 objective를 exact resume하는 요청은 거부한다. raw graph cache는 재사용하지만 새 loss의 학습은 별도 output에서 시작한다. 외부 GitHub 업로드와 서버 학습은 수행하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 시 모델 축소보다 메모리 원인을 우선한다. 이번 실제 검사는 OOM 없이 완료했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 합성 단위 검사는 실제 CT 검사와 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
