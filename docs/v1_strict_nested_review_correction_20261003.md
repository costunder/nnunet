# v1 그래프 크기 대조의 strict-nested 수정

2026-10-03 독립 검토를 반영한다. 이전 `HierCP_v1_GRAPH_FIRST_GPT_REVIEW_20261003.zip`, canonical-resample sampler와 104/416 결과는 보존한다. 이번 변경은 v1의 모델이나 GT를 v2.2로 바꾸는 작업이 아니다.

## 발견한 혼입과 변경 범위

이전 작은 그래프는 canonical node pool에서 새로 선택했다. 실제 원본 sampled view 밖에서 들어온 context node가 104-seed에서1,809개,416-seed에서6,243개였다. 모델/GT/loss는 같아도 sampling distribution까지 달라졌으므로 ‘원본 그래프를 줄였을 때의 효과’로 단독 해석할 수 없었다.

새 `nested_graph_size.py`는 원본 `build_local_view()`가 만든 **그 sampled graph**를 입력받는다. Seed·role/shell FPS·관계 witness·relay 모두 해당 graph 내부에서만 찾는다. 새로운 canonical node, kNN, 반경 edge, 평균 supernode를 넣지 않는다.

```text
canonical cache
  → 원본 v1 build_local_view (native sampling 그대로)
    ├─ original-v1 대조
    └─ native view 내부의 role/shell seed + 원래 경로 relay
        → 선택 node의 induced directed edges와 기존10D attrs
        → nested-small-v1 대조
```

`Vsmall⊆Vnative`와 `Esmall=Enative|Vsmall`을 검사한다. 실제 sampled index와 canonical full_id를 구분하며 node features/좌표/edge attrs/order/중복 edge multiplicity를 그대로 보존한다. Native view는한 번 만들고 두 대조가같은view를 공유한다. 다만 role별 FPS로 노드를 삭제하는 편향까지 없어지는 것은 아니다. 이는 canonical-resample 혼입을 제거한 대조이며 임의의 정보 손실이 없는 압축이라는 주장이 아니다.

104/416은 기존 명시적 **seed profile**을 그대로 시험한다. Relay를더하면최종N은더커진다. 최적값이나최종node/edge상한으로두지않고, coverage를PASS시키려고 seed·radius·hop·budget을자동변경하지않는다.

## 정보 전달 감사

`graph_flow_audit.py`는 원본전체source/원본에서선택된source/nested source를분리해검사한다. CUDA에서 source-role 집합을벡터화해 실제src→dst방향의 누적1/2/3-hop도달성을계산한다. 원본3-layerGAT와같은hop수로경로손실을본다.

- tumor_surface→source_context/target_context
- source_context→target_context
- source/target context↔각liver_surface
- tumor_interior↔tumor_surface

Selected target분모와원본target분모를함께기록하며 원본선택source로도달하던선택target이nested에서사라지는비율을따로본다. Source를삭제한영향과중간경로를삭제한영향을섞지않는다. 이지표는source집합에서대상까지의**union**이지모든개별node쌍의정보보존률은아니다.

Role별zero-in-degree와relation별degree histogram,role내/전체weak components,context shell별coverage,원본→선택nearestdistance,선택→원본nearestdistance,directed tumor→context shortestpath,source↔target correspondence endpointcoverage도기록한다. 전자의거리값은공간coverage이고후자는strictsubset이면0이어야하는identity검사다. Weak connectivity를3-hopdirected정보보존으로승격하지않는다.

## 변경하지 않는 계약

원본CNN5채널48³/base12/output32,hidden128,GAT3×4heads,role/shellattention,pairfusion,L1/L2/scalarhead,source-anchorGT,curriculum/loss,두view,후보pool128/학습8candidates,원본전체paste mask를유지한다. BasicCP와v2.2의observedP/U정답을변경하지않는다. `U=CP부적합`또는`P=외부donor와의적합정답`으로해석하지않는다.

`group_search.py`계획포맷v2는quality(local_features/local_operator)와runtime(activation_storage)를분리한다. 두도메인의자동혼합과구v1계획의자동재해석을거부한다. L0readout/fusion은추후같은128Dinterface의품질대조후보이며현재조합registry에구현완료로등록하지않았다. v2.2 L1은operator/node/edge semantics가함께바뀌는compositearchitecture,prototypecosinescorer는representation/prototype/scoring이바뀌는별도실험이다.

## 실제 검사 범위

실행기는 `tools/verify_v1_nested_graph_size_debug.py`이다. Actual CT/fixture/source SHA를검사한뒤원본과nested의CNNpatch·candidatecenter·difficulty·corruption·상위graph를exact equality로대조한다. 실제모델을CUDA에서원본seed42같은state로초기화하고원본loss/gradient/clip/AdamW/held-out scoring/full-maskpaste까지실행한다. 별도의production checkpoint나ready표시는만들지않는다.

디버그코호트는train liver5/6,held-out liver31이다. 모델은10,434,532parameters 그대로이며physical2samples×8candidates=16graphs/view를disjoint-unionbatching하고두view를사용한다. 2update는기계적smoke이며서버전체학습이나과거v1의성능재현이아니다.

Original materialization/nested thinning/directed audit/complete compute update비용을분리한다. Compute는동기화된forward/loss/backward/gradient검사/clip/optimizer이며loader,H2D,preparation,전체validation,checkpointIO는제외한다. 최초forward평가는warm-up효과가있지만첫backward/optimizer할당도측정값에포함하므로steady-stateepoch배속으로환산하지않는다.

## 실제 결과: 실행 검사는 통과했고 정보 전달 손실은 남았다

단위 검사는 총73개가 통과했다. 이 중68개는 입력·topology·계획·파일 결속 metadata 검사이고,5개는 CUDA directed-flow 검사다. 추가로 실제 CT3case에서 각 프로파일의48view 전부를 감사하고, 원본/작은 그래프 각각2update의 neural forward → 원본 loss → backward → gradient 검사 → AdamW 갱신 → held-out scoring → 전체596voxel mask paste를 실행했다.

| 항목 | 원본 sampled v1 | nested104 | nested416 |
| --- | ---: | ---: | ---: |
| 평균 node 수 | 11,335.21 | 350.69 | 1,024.19 |
| 평균 typed edge 수 | 783,205.85 | 3,140.77 | 18,303.77 |
| 같은 native view 밖의 node | 해당 없음 | 0 | 0 |
| 평균 relay 수 | 해당 없음 | 246.69 | 614.52 |
| native weak component 일부 미선택 view | 해당 없음 | 2/48 | 0/48 |
| isolated node를 포함한 view | 비교 원본 참조 | 16/48 | 5/48 |
| compute update 평균 | 각 대조에서40.03 / 38.53초 | 3.44초 | 2.63초 |
| 최대 GPU allocated | 6.29GiB | 0.48GiB | 0.63GiB |

원본과 작은 그래프의 모델은 모두10,434,532parameters다. 각 프로파일 안의 원본 대조와 비교하면 compute 시간은 각각 약11.65배/14.63배 차이였다. 이 값은2update의 제한된 계산 구간이며 전체 epoch 배속이 아니다. 과거 대조의22초 기준과 새 대조의38~40초 기준을 섞어 계산하지 않는다. 실제 비용 변동의 원인을 별도 profiler로 확인한 결과도 아니므로, 작은 그래프 두 프로파일의2step 평균 차이를 최적값 선택 근거로 삼지 않는다.

48view 준비 비용은 native materialization이 각각9.76/9.51초, 그 이후 nested thinning이5.27/6.00초, 전체 directed audit가15.17/14.76초였다. 같은 native view를 공유하는 대조군을 만들기 위해 먼저 원본 view의 준비 비용을 낸다. 생산 학습에서 이 정적 결과를 얼마나 재사용할지와 loader/validation/save 전체 비용은 이번 smoke로 측정하지 않았다.

원본에서 **선택된 출발 노드로3hop 이내에 도달하던 선택된 도착 노드**를 분모로 한 경로 손실은 다음과 같다. 각 view의 대상 수를 합친 weighted 비율이다.

| 방향 | nested104 손실 | nested416 손실 |
| --- | ---: | ---: |
| tumor surface → source context | 152/4,153 (3.66%) | 594/12,767 (4.65%) |
| tumor surface → target context | 132/2,860 (4.62%) | 466/9,599 (4.85%) |
| source context → target context | 146/2,920 (5.00%) | 143/9,758 (1.47%) |
| source liver surface → source context | 307/3,262 (9.41%) | 884/9,820 (9.00%) |
| target liver surface → target context | 382/2,539 (15.05%) | 1,018/8,411 (12.10%) |

두 프로파일의 선택 노드 집합과 분모가 다르므로 이 표만으로104/416의 정보 보존 우열을 결정하지 않는다. 삭제된 원본 target까지 포함한 coverage와 source 삭제 영향은 각 raw flow report의 별도 분모에 기록돼 있다. 이 숫자는 전체 원본 정보 손실률이나 암 특징 손실률이 아니다. **약한 연결성분의 연결을 보존해도3-layer GAT의 방향별 문맥 경로가 전부 남지는 않았다는 실제 반례**다. 결과를 맞추려고 seed나edge를 자동 추가하지 않았다.

2update 후 train2case MRR은 original 대조1.0, nested104 1.0, nested416 0.75였다. held-out1case는 original 약0.143, 두 nested 프로파일 약0.167이었다. 작은 분모와2update fitting으로 추천 정확도 유지·일반화 개선을 주장하지 않는다. 테스트가 통과한 것은 실행 연결과 불변성이고, 품질 유지 평가는 별도다.

원시 결과는 `work/v1_nested104_CUDA_DEBUG_20261003/`와 `work/v1_nested416_CUDA_DEBUG_20261003/`의 report/sampling_audit/flow_audit/branch JSON이다. `validation/v1x_progressive_20261003/nested_verification.json`에 source·원시 report SHA와 검사 범위를 기록했다. Flow와execution contract 파일을 다른 프로파일 것으로 바꾸거나 측정 이후 sampler 코드를 바꾸면 전달본 검사가 거부하는3개 회귀 검사도 통과했다.

## 다음 학습 판단

먼저original-v1과strict-nested-small-v1만동일84train/21validation분할에서비교한다. Seed42,원본anchorGT,8candidatecurriculum/difficulty/corruption,train-onlyprototype,두view,동일초기state/optimizersteps/loss/physical+effectivebatch/fixedvalidation을결속해야한다. 과거105/26분할의높은점수를84/21에서재현했다고부르지않는다.

Train곡선에는native총loss/pairloss/MRR/positive-bestothermargin을,held-out에는21case의pairedMRR/top1/margin과casebootstrapCI를기록한다. 허용품질차이는미정이므로기본값을만들지않는다. 먼저한seed에서같은조건을비교하며추가seed는판단이애매할때의후속실험이다. 모든조합40epoch를자동실행하지않는다.

128candidateCP추천/fullmask검사와nnU-NetBasicCP80비교는순위품질과실행조건이확인된뒤별도진행한다. 이번로컬검사는장기학습을자동시작하지않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 사용자가 요청한 그래프 크기 대조는별도profile이다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. DEBUG에서는 같은physical2sample/16graph를두대조에유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. RTX5070Ti16GiB/CPU16logical·8physical/RAM63.93GiB/명시적CUDA12GiB·RSS32GiB예산을확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번대조에서자동축소는없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제CT·CUDA원본모델4branch를검사했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
