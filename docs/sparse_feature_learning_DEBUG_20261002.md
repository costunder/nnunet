# v2.2 희소 L0 — 노드 수별 실제 연속 학습과 별도 CT 평가

2026-10-02. 기존 48/96/192의 coverage·fresh update 검사 다음에 추가한 **짧은 DEBUG 학습 비교**다. 이전 코드·측정·3D 시각화는 보존했다. 최종 production 노드 수나 최적 그래프 크기를 승인한 결과가 아니다.

## 확인한 결과

같은 초기 가중치·관측 목록·update 순서에서 각 설정을 **60회 연속 AdamW update**했다. 매번 초기 모델로 되돌리는 비용 검사를 순위 학습으로 취급하지 않았다. 20회마다 현재 CNN으로 train support 전체를 갱신한 뒤, 현재 query CT를 다시 인코딩해 학습 CT와 독립 CT를 평가했다.

| Context nodes | Train P/U pair-win: 초기→60update | 별도 CT pair-win: 초기→60update | 별도 CT MRR: 초기→60update | 별도 CT R@5: 초기→60update | Update 중앙값, 초 | Peak 할당, GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 39.06%→91.35% | 70.51%→43.26% | .3333→.0588 | .125→.000 | .718 | 4.394 |
| 96 | 37.19%→92.40% | 73.54%→48.44% | .3333→.2000 | .125→.125 | 1.012 | 4.396 |
| 192 | 36.09%→94.79% | 73.05%→54.88% | .3333→.2000 | .125→.125 | 1.595 | 4.396 |

Pair-win은 `s(P)>s(U)`인 쌍의 비율이며 동점은 0.5로 센다. **분류 정확도나 CP 적합성 정확도가 아니다.** MRR는 각 CT의 첫 관측 종양 순위의 역수다. 별도 CT는 P8/U128인 liver_31 한 개이며, 최종 첫 P의 순위는 48에서17위, 96/192에서5위였다. 초기에는 세 설정 모두3위였다.

Train pairwise ranking loss는 .6966→.2756 / .6941→.1916 / .6932→.1480으로 줄었고 P−U 평균 margin은 −.0051→2.3328 / −.0002→3.3969 / .0015→3.6646으로 벌어졌다. 반면 별도 CT loss는 .6792→1.2454 / .6779→1.2380 / .6781→1.1364로 악화했다. 단순 공통 점수 이동만 관찰한 결과가 아니다.

**이 DEBUG cohort에서 훈련 관측의 순위 분리는 학습됐지만, 별도 CT의 순위 개선은 확인되지 않았다.** 192가 같은 종료 step에서 48보다 높은 pair-win을 보인 것은 사실이다. 그러나 자기 초기값 대비 별도 CT의 MRR·pair-win·loss가 악화했고 96과192의 R@5도 같았다. 따라서 48의 충분성, 96/192의 최적성, ‘노드 수를 늘리면 일반화 문제가 해결된다’는 결론을 내리지 않는다. 이 작은 cohort의 적합/일반화 차이가 기존 서버 학습 부진의 원인이라고 확정하지도 않는다.

학습·검증 곡선: [learning_curves.png](../validation/l0_sparse_feature_learning_DEBUG_20261002/learning_curves.png). 원시 결과와 정확한 집계는 같은 폴더의 `report.json`, `compact_summary.json`이다.

## 데이터와 정답을 유지한 범위

- **P=원본 CT에서 관측된 적격 종양 anchor, U=원래 미관측 비교 위치.** U를 CP 부적합으로, P를 특정 donor와 호환되는 CP 정답으로 재정의하지 않았다.
- 기존 로컬 DEBUG의 case 구분을 그대로 사용했다. Train은 liver_66/71/72/75, validation은 liver_31이다. 각 case에서 원본의 모든 P와 U128을 복원했다. 기존 DEBUG의 P1/U1만 가지고 graph-size 효과를 비교하지 않았다.
- Train은 P15+U512=527관측, validation은 P8+U128=136관측이다. 전체 원본14,102관측의 **663개, 4.70%인 명시적 DEBUG cohort**다. 전체84 train/21 validation 실험을 실행하거나 대체한 것이 아니다. Production 데이터 목록·split·40epoch 설정은 보존했다.
- 원본 ID·좌표·target·patient identity·P component/anchor multiset·U center multiset·U128 개수를 검증했다. 후보 위치나 anchor를 새로 생성하거나 이동하지 않았다.
- 각 case의 donor는 기존 DEBUG의 train-only pool 배정을 고정했다. Train support에서 query 환자의 recipient 및 donor 항목을 제외한다. Validation CT는 support·teacher fitting·optimizer update에 들어가지 않는다.
- Basic CP80%, seed42, 원본 paste mask와 전체 mask 검사 경로는 변경하지 않았다. 이 실행에서는 segmentation 및 CP 효용 평가를 하지 않았다.

## 고정한 모델·학습 경로

CNN8Conv `[12,24,32]` → 현재 CNN 특징/상대 위치/role의 희소 그래프 → mean GraphSAGE3층128D → query/near/mid/wide readout → 기존 donor-recipient fusion128D → 기존 L1/L2 → 기존 `same_donor_live_v1` 전체 loss → backward → clip → AdamW다.

Context quota만 band당16/32/64로 달라져 context48/96/192, abstract query까지 scene49/97/193이다. 기본 radius/query/edge 규칙은 이전 비교 그대로다: query3mm, near≤5mm, mid(5,10]mm, wide는 원래 margin10 crop의 나머지, spatial3NN+feature1NN symmetric union. 모든 sampled context node는 원본 간 안에 있다. Query는 이동하지 않은 원본 anchor의 추상 readout이며 실제 voxel sample로 주장하지 않는다. Hard selection index는 미분되지 않지만 선택된 현재 CNN 특징을 다시 gather하여 CNN까지 gradient를 전달한다.

모든 설정의 전체/trainable parameter 수는 **1,242,198**로 같다. 초기 모델 hash는 모두 `6bd04b2259a547967e27ee90eeba81344e93bfddeb525fb72450c0f37160b047`이다. CNN과 paired fusion, L1/L2는 무결성을 검증한 로컬 DEBUG saved step4에서 복사했다. Graph/readout은 seed42 새 초기화다. 서버에서 40epoch 학습한 모델이나 checkpoint exact resume 비교가 아니다.

각 cycle은 전체 P×U **1,920쌍을 정확히 한 번씩** 비교하는 기존 LiveContext20 tiles다. 527개 관측을 모두 사용하며 반복되는 P의 CE는 기존 multiplicity 보정을 유지했다. Cycle당 query presentation은587개다. 세 설정은 같은 원본 ID schedule을 공유한다.

설정 physical batch32/effective32, accumulation1을 유지했다. 실제 cycle에는 full32 tile16개와 자연스러운 마지막 tile25/20/15/15가 있다. 이를 숨기거나 메모리 때문에 physical batch를 낮춘 것으로 보고하지 않는다. Optimizer는 설정별 한 번만 만들며 모든 parameter의 Adam step이1→60으로 연속 증가하는지 검사했다. 평가 때 모델·buffer·Adam 상태 hash가 그대로이고 RNG·mode가 복구되는지도 검사했다.

LR1e−4, weight decay1e−4, grad clip5와 기존 loss/multiplicity는 유지했다. 실제 DEBUG precision은 FP32, autocast/TF32 off, deterministic이다. AdamW backend는 세 설정 모두 Torch2.8 기본 `fused=None, foreach=None`이다. 원본 production 설정의 `fused_optimizer=true`/AMP와 동일한 실행 backend라고 주장하지 않는다. 이 차이는 세 설정에 공통이며 원본 설정은 변경하지 않았다. 원본 설정과 실제 backend는 `publication_receipt.json`에 구분했다.

Detached L0 support는 cycle 안에서 고정하고, 각 case 첫 tile의 현재 L1으로 support-only teacher를 fit한다. L1/L2의 현재 학습 연산을 cache로 고정하지 않는다. Cycle 종료 후 현재 L0로 train527 전체 memory를 다시 만들고 그 memory를 다음 cycle에 재사용한다. Evaluation의 query도 현재 CNN으로 다시 계산한다. 처음의 query embedding이나 이전 cycle의 support basis로 평가하지 않는다.

## 실제 자원과 비용

RTX5070Ti16GiB 한 개, CPU16 logical, 시작 가용 RAM42.33GiB. 명시적 CUDA12GiB/RSS32GiB/resident12GiB 한도, reader workers4와 한 batch 앞의 CPU producer prefetch를 사용했다. GPU에서 scene을 batch로 계산하며 sample마다 CNN/GNN forward를 따로 돌리지 않았다.

첫 train-memory physical32 입력은 `[33,1,58,50,7]`이었다. 세 설정의64 scene은 각각49/97/193 nodes, directed edge 범위252–312 /492–556 /910–1014다. 첫 실제 batch 수치이며 전체 cohort의 edge 최대값으로 주장하지 않는다. Shape는 native crop와 batch padding 결과이며 CT 공간 해상도를 바꾼 것이 아니다.

| 항목 | 48 | 96 | 192 |
|---|---:|---:|---:|
| 60update 계산 중앙값, 초 | .718143 | 1.012479 | 1.595209 |
| Cycle3 전체 비용: update+refresh+평가, 초 | 33.604 | 54.982 | 83.785 |
| 세 cycle 전체 비용, 초 | 101.371 | 155.223 | 253.466 |
| Initial train-memory 생성, 초 | 59.171 | 12.248 | 20.645 |

Update 측정은 GPU에 입력을 보낸 뒤 CNN/노드 선택/GNN/L1/L2/loss/backward/finite·gradient 검사/clip/AdamW까지다. Loader 대기·전송·case teacher 준비는 별도로 기록했다. Cycle 비용에는 이 작업들과 현재 support refresh 및 train/validation 평가가 포함된다. 임시 JSON 출력·초기 원본 CT 읽기·production checkpoint 저장은 포함하지 않는다. **서버 전체 epoch의 예상 시간으로 환산하지 않는다.** 첫48 initial-memory는 원본 CT cold read가 포함되고96/192는 동일 RAM crop/raw cache를 재사용하므로 initial-memory 값으로 배속을 주장하지 않는다.

최종 RSS7.12GiB, raw cache4.80GiB와 crop cache68.05MiB, cache hit19,615/miss668이었다. Peak VRAM은 그래프 크기에 따라 고정한 값이 아니라 실제 배치에서 측정한 할당값이다. 현재 입력에서는 CNN 계산의 메모리가 커서 node quota를 늘려도 peak가 거의 같았고, update 시간은 증가했다. 원본 CT·mask의 새로운 대형 캐시는 생성하지 않았다.

## 검사·보존·미검증

- 단위/기존 회귀 **32개 PASS**. CPU metadata/계약 검사는 모델 추론으로 보고하지 않는다. 수치 경로와 실제 CT 학습·평가는 CUDA에서 실행했다.
- 실제180update 모두 loss/전체 gradient finite, 7개 모듈 gradient>0, 연속 Adam step, 각 cycle의 7개 모듈 실제 parameter delta>0를 확인했다. Role embedding을 포함한 모든 trainable parameter의 gradient 존재도 검사했다.
- 독립 agent의 코드 감사에서 supervision·continuous optimizer·memory refresh·support exclusion·fair schedule·evaluation state 보존의 material defect를 찾지 못했다. 지적된 Adam backend 차이는 실제 설정으로 명시했다.
- 기준 production121파일, 입력 assignment, source checkpoint, 측정 구현 hash, 이전3D 시각화와 기존 결과 보존을 검사했다. 관계없는 수정 중인 `REFERENCES.md`, `code.txt`, `gpt_handoff.md`도 그대로 보존했다.
- 구현/정적/단위/실제 CUDA 짧은 연속 fitting은 완료했다. **전체 학습·전체 validation21case·segmentation·CP 효용·그래프 최적 크기·기존 서버 모델 대비 정확도 개선은 미검증**이다. Production checkpoint·ready 표시는 생성하지 않았다.
- 한 validation CT의1,024 P/U쌍은 독립1,024 case가 아니다. 그 쌍으로 일반화 통계의 표본 수를 부풀리지 않는다. 종료60step은 사전 고정했고 validation에서 유리한 step을 골라 best 성능으로 제출하지 않았다.

## 재실행 가능한 짧은 DEBUG 경로

기존 파일 덮어쓰기를 거부하므로 output은 새 경로여야 한다. 아래는 같은 로컬 데이터가 있는 작업공간에서 재현하는 DEBUG 명령이며 서버 전체 학습 launcher가 아니다.

```powershell
.\.venv\Scripts\python.exe -u -B tools/compare_sparse_feature_learning_debug.py `
  --run work/local_cnn_experiment_resume_DEBUG_20261001/resumed `
  --assignment work/v222_v1_full_training_20260924/cache/pair_assignment.json `
  --assignment-receipt validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json `
  --output work/sparse_feature_learning_CT_DEBUG_20261002_repeat `
  --debug-cycles 3 --query-radius-mm 3 --near-radius-mm 5 --mid-radius-mm 10 `
  --workers 4 --cuda-gib 12 --rss-gib 32 --resident-gib 12
```

Source checkpoint SHA256: `9b36e85a57819042026b152e7bcc7236ed9ad0b372fa32b5d5c321d74b50c1f6`.
Original assignment SHA256: `b7347a7d383f7010a6c114874121f8a2bb60d3910dd807ed4d6e4de0f433a19f`.
원시 metric JSON·unit log·출처 receipt·metric figure는 `validation/l0_sparse_feature_learning_DEBUG_20261002/`에 보존했다. Native CT·mask·weights·anatomy payload는 Git에 공개하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 기존 명시적 DEBUG case에서 모든 원본 P/U128을 복원했으며 production 데이터는 보존했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 명시적 DEBUG cohort/3cycle 범위를 원시 기록과 문서에 표시했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. Batch32, 자연스러운 tail, batched scene, 병렬 raw reader/prefetch를 기록했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행에는 OOM이 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 새 가중치 초기화와 실제 학습을 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
