# v2.2 — v1 문맥 관계를 반영한 희소 L0, 실제 CT DEBUG 비교

2026-10-02. 사용자 요청은 v1의 donor–recipient 관계 구성과 역할별 문맥 집계를 참고하되, 이전의 큰 그래프를 그대로 복원하지 않는 것이다. 이번 별도 후보는 실제 입력부터 전체 loss·backward·optimizer까지 구현했다. **실제 CT에서 연결 분리가 확인되어, 완성된 추천 모델이나 production 학습 가능 상태로 표시하지 않는다.**

## 구현과 원래 v1의 차이

```text
원본 donor/recipient 간 내부 native 국소 crop
 → 기존 8-convolution CNN [12,24,32], strides 1/2/4
 → 간 내부 위치의 현재 multi-scale CNN 특징 68D
 → near/mid/wide의 공간+CNN 다양성 대표점 선택
 → donor query/near/mid/wide + recipient query/near/mid/wide
 → 같은 pair 안의 8종 방향 관계, 관계별 평균을 분리
 → 3층·128D 관계별 mean GraphSAGE + 자기 residual 1회/층
 → 8역할의 별도 value transform + 6문맥 역할의 attention pooling
 → donor128D / recipient128D
 → 기존 [d,r,r−d,r*d] fusion → L0 128D
 → 기존 L1 / L2 / 전체 objective / backward / AdamW
```

Donor와 recipient를 독립 인코딩한 뒤 마지막 벡터에서만 결합하던 경로와 달리, 이번에는 **donor의 메시지가 fusion 전에 recipient 노드 표현에 도달**한다. GPU 단위 검사에서 donor CT만 바꾸어 recipient joint hidden/readout의 변화를 확인했고, 다른 pair로 영향이 번지지 않음을 확인했다.

v1의 실제 활성 구현은 `hiercp/spatial.py`다. v1은 역할별 공간 cell 표본, 필수 interface 이웃, seed 표본 이후 2-hop 확장으로 fine 이웃을 보존하고 radius 관계를 만든다. 이번 후보는 **그 필수 이웃·hop 확장을 복원하지 않았다.** 이전 sparse selector가 고른 대표점에 v1의 context/interface/correspondence 반경 6/8/5mm와 관계별 처리·역할별 readout을 적용한 구조다. v1의 GAT·종양 interior/surface 노드·handcrafted 통계 특징을 복원한 모델이 아니다.

## 노드와 edge의 정확한 계약

Margin10mm인 동일 native crop을 사용한다. Query는 원래 anchor 반경3mm 안의 organ-supported CNN 특징을 평균낸 **추상 token**이다. 특정 종양 내부 voxel·혈관 시작점으로 정의하지 않는다. Near는5mm 이하, mid는5–10mm, wide는10mm 밖이면서 기존 crop 안이다. 각 band의 quota16/32/64가 이번 DEBUG 비교의48/96/192 문맥 노드에 해당한다.

| 관계 | 반경 | 받는 노드당 최대 source 수 |
| --- | ---: | ---: |
| donor context → donor context | 6mm | 3 |
| recipient context → recipient context | 6mm | 3 |
| donor query → donor context | 8mm | 3 |
| donor context → donor query | 8mm | 3 |
| recipient query → recipient context | 8mm | 3 |
| recipient context → recipient query | 8mm | 3 |
| donor context → recipient context | 5mm | 1 |
| donor query → recipient context | 8mm | 3 |

반경 조건을 먼저 적용하고 그 안의 가까운 source를 고른다. `k`는 receiving node당 source 상한이다. Singleton query 하나는 반경 안의 여러 context로 나갈 수 있다. Cross 관계는 양쪽 **원래 anchor 기준 상대 physical-mm 좌표**를 비교한다. 환자 간 해부 구조 등록을 수행한 좌표나 실제 혈관 연결이 아니다. 반대 방향 cross 관계를 추가하지 않는다.

역할·거리 조건을 벗어난 edge, feature-only 연결, MST, 추가 hop, 연결 부족 시 반경 확대·sample skip·노드 drop·fallback을 넣지 않았다. 선택 index는 비미분적이고 선택된 live CNN 특징을 다시 gather하여 영상 encoder로 gradient를 전달한다. CNN은 unique crop당 한 번 실행한다. 환자 전체 CT의 CNN1회 재사용을 구현했다는 뜻은 아니다.

Node 수는 quota에 미달하는 nonempty band에서 줄어들 수 있으므로 항상 실측 mask 수를 기록한다. 이번 case의 실제 pair 크기는98/194/386이다. Query/band가 없거나 입력 무결성이 틀리면 명시적으로 실패한다. 간 밖/padding CT 값의 NaN 변경이 결과에 영향을 주지 않는 CUDA 검사를 통과했다.

Message passing은 관계별 sparse COO/SpMM을 사용하고 별도 root transform을 중복 적용하지 않는다. **그래프 생성에는 `[B,N,N]` 거리와 8종 masked sort가 남아 있다.** 전체 forward를 선형 시간 알고리즘으로 부르거나 모든 dense tensor가 사라졌다고 주장하지 않는다.

## 실제 CT에서 확인된 연결 분리

Liver66의 원래 P5+U128 모두를 같은 donor liver1/component1로 검사했다. 같은 quota의 두 경로는 fine pool, 선택 CNN 특징, native/mm 좌표와 mask가 정확히 같았다. 각 quota의 선택 집합은 nested이고 전체 유효 위치의 거리 진단이 감소하는 것을 확인했다. 이 거리는 암 관련 정보 보존율·추천 정확도가 아니다.

| 문맥/branch | Pair nodes | 새 관계 edge/pair | Cross edge/pair | Weak components/pair | 고립 노드/pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| 48 | 98 | 245–289 | 47–60 | 19–32 | 13–27 |
| 96 | 194 | 589–682 | 102–124 | 21–44 | 12–36 |
| 192 | 386 | 1,338–1,492 | 214–264 | 19–53 | 10–35 |

모든133 pair에 cross 관계가 존재하고 역할/반경 위반은0이었지만, **전체 문맥이 하나로 연결되지는 않았다.** 대표점을 공간·CNN 다양성으로 넓게 고른 뒤 작은 물리 반경만 적용하므로, 사이의 연결을 이어 줄 fine 이웃을 보존하지 않은 문제가 드러났다. 192개로 늘려도 분리 성분이 남는다. 단순히 노드 수를 늘리는 것만으로 해결됐다고 할 수 없다.

분리된 노드도 역할별 readout에는 참여하고 gradient를 받는다. 하지만 그 사실은 해당 노드가 다른 문맥과 message passing으로 상호작용한다는 증거가 아니다. 따라서 전체 path의 기계적 검사 PASS와 **문맥 그래프의 연결 설계 한계**를 구분한다. v1의 필수 이웃/hop 보존과 이번 대표점 선택의 차이를 다음 설계에서 해결해야 한다. 이번에는 측정 결과를 PASS로 만들기 위한 반경 확대나 가짜 bridge를 추가하지 않았다.

표시한24개 실제 joint graph에 대해 별도 BFS를 계산하여 GPU와 export의 성분 수가 모두 같음을 확인했다. 두 query가 속한 weak component 밖의 노드는 pair당 평균29.375/48/74.625개였고 주로 wide 영역이다. 실제 방향으로3층 역추적하면 recipient query까지 메시지가 도달할 수 있는 노드는19–26/21–31/17–28개이며, 표시된24그래프에서 wide node는 그 query에 도달하지 못했다. **Weak connectivity와 방향·layer 수에 따른 message reachability도 서로 다른 조건**이다. 역할별 pooling이 모든 노드를 직접 읽는 경로는 별도로 남아 있다. 원래 v1이 모든 경우에 완전히 연결됐다는 주장도 하지 않는다.

## 같은 작업량의 계산 비용

RTX5070Ti16GiB, FP32, TF32/autocast off, deterministic on. CPU8physical/16logical, RAM약64GiB와 시작 가용42.21GiB 확인. Explicit CUDA12GiB/RSS32GiB/resident12GiB, parallel reader4, pinned/nonblocking 전송, raw/crop cache 재사용을 기록했다. 단일 할당 GPU의 comparative DEBUG다.

동일 physical/effective32, accumulation1의 P5+U27 tile에서 기존 ranking135쌍·CE·alignment·전체 정규화를 유지했다. 동일 global loss context139관측/8step schedule에 연결하되 한 tile의 **연속3update**만 측정했다. 각 branch는 disposable warmup1회 후 모델·buffer·RNG를 복원하고 fresh AdamW를3회 이어 사용했다. Adam state step3과 모든 trainable parameter의 optimizer 포함을 확인했다.

| 문맥/branch | 이전 독립 그래프 update 중앙값 | 새 관계 그래프 update 중앙값 | 새 경로 peak allocated |
| --- | ---: | ---: | ---: |
| 48 | 0.514s | 0.463s | 2.207GiB |
| 96 | 0.650s | 0.641s | 2.207GiB |
| 192 | 1.004s | 1.031s | 2.207GiB |

Update는 CNN·대표점 선택·edge 생성·L0/L1/L2·전체 loss·backward·finite/clip·optimizer를 포함한다. 이미 메모리에 올린 동일 batch의 측정이며 loader/transfer·진단용 parameter CPU 복사·실제 production checkpoint 저장은 제외된다. Peak는 **6개 비교 모델이 동시에 있는 process 전체 CUDA allocated**이고 isolated model의 최대 batch admission 측정이 아니다. 서로 다른 모델 parameter 수는 이전1,242,198, 새1,785,558이다.

새 경로의 두 번째 측정 update에서 CNN/선택/edge 생성/3층 message 시간은 각각48일 때0.0098/0.1943/0.0089/0.0030초,96일 때0.0105/0.3623/0.0143/0.0027초,192일 때0.0103/0.7285/0.0239/0.0047초였다. **이 작은 그래프 검사에서는 대표점 선택이 남은 큰 연산이다.** 192 설정에서 새 관계 경로가 더 빠르다고 주장하지 않는다.

저장된 DEBUG memory8관측을 각 모델로 다시 인코딩했고, 원래 recipient/donor group 제외 후 eligible support6 ID가 모든 경로에서 같았다. Teacher를 재구축하여 기존 CNN-mean memory와 섞지 않았다. 3update 동안 자기 branch의 detached support/teacher를 고정했으며 이후 refresh 비용을 별도 측정했다. 새 경로 refresh는48/96/192에서0.264/0.350/0.703초였다. **전체 production support11,279개·validation·checkpoint·epoch 비용 측정이 아니다.** 3시간/epoch 해결로 환산하지 않는다.

CNN/fusion/L1/L2는 무결성을 확인한 로컬 DEBUG step4 snapshot에서 정확히 복사했다. Graph/readout은 seed42 새 초기화다. 실제 서버 장기 학습 가중치나 이전 checkpoint의 exact resume가 아니다. 새 L0가 v1보다 낫다거나 CP 성능이 오른다는 검증은 없다.

## 검사와 보존

- 정적 검사 완료. 별도 L0·기존 sparse helper·coverage 검사27개 PASS: profile metadata1개, numerical CUDA fixture26개. Fixture는 unit 입력이고 실제 CT 성능 데이터로 부르지 않는다.
- 실제3D 화면17검사 PASS, browser errors0. 세 노드 밀도 동시 표시·같은 mm 축척·P/U 전환·관계 필터·방향선·회전·노드 원본 좌표·원본 GT contour 토글·mobile을 확인했다. 저장 상태가 되돌아올 때 선택 노드가 초기화되던 UI 문제를 수정했고, GT contour가 edge에 가려지지 않도록 실제 모델 geometry를 바꾸지 않고 화면 레이어 순서만 보완했다.
- 실제 CT133관측×6경로 전체 forward, 동일 selected feature/좌표, 원본 GT/geometry 결속, own support/teacher 재구축 완료.
- 6경로×3연속 update의 finite gradient·모듈별 실제 parameter 변화 확인. 새 모델의3층×8관계 weight,8역할 value,6문맥 attention scorer 모두 실제 loss gradient와 delta가 있다.
- 기존 Basic CP·L1/L2 구조/수식·loss·P/U 정답·전체128후보·원본 paste mask 검사는 변경하지 않았다.
- 기존 checkpoint·assignment·memory·loss schedule·production source를 검사 전후 비교했다. 새 구현/실행 helper8개 SHA도 시작/종료에 같음을 확인했다.
- 기존 production121파일과 작업 전 dirty3파일의 SHA를 최종 대조하여 보존했다. `verification.json`, `browser_checks.json`, `independent_connectivity.json`에 실행 범위와 숫자 근거를 기록했다. CPU 독립 BFS는 저장 graph의 통계 감사이며 CPU 모델 forward/학습이 아니다.
- 처음 실행은 계산 후 새 출력 directory 생성의 Windows 접근 오류로 결과 저장에 실패했다. 이를 숨기지 않고 output exclusive 생성·started receipt를 실행 초기에 두고, branch별 완료 JSON을 별도 저장하도록 수정했다. `r1`에서 전체 재실행 및 결과 저장 완료. 앞선 실행의 loaded/source hash 혼동을 최종 근거로 사용하지 않는다.
- 전체 학습/전체 평가/production ready/checkpoint 작성 없음. 실제 CT 비교의 그래프·주석 payload는 로컬 시각화에만 사용하고 공개 Git에는 CT pixels/mask/weights를 넣지 않는다.

Raw detailed report는 `work/relational_sparse_CT_DEBUG_20261002_r1/report.json`에 보존했다. 약97MB이므로 공개 기록에는 그 SHA와 모든133 record별 연결 수, 전체 measured update 및 수치 검사를 담은 `validation/relational_sparse_20261002/actual_CT_DEBUG_summary.json`을 사용한다. 화면의 좌우 offset은 display용이며 원본 모델 좌표·edge를 변형하지 않는다.

## 로컬 재현 명령

이 명령은 현재 로컬 입력이 있는 workspace에서 쓰는 DEBUG 명령이다. 기존 출력 directory가 있으면 거부하며, 새 output 이름을 명시해야 한다. 장기 학습 실행 명령이 아니다.

```powershell
.venv\Scripts\python.exe -u tools/verify_relational_sparse_ct_debug.py `
  --run work/local_cnn_experiment_resume_DEBUG_20261001/resumed `
  --assignment work/v222_v1_full_training_20260924/cache/pair_assignment.json `
  --assignment-receipt validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json `
  --case liver_66 --output work/relational_sparse_replay_DEBUG `
  --query-radius-mm 3 --near-radius-mm 5 --mid-radius-mm 10 `
  --workers 4 --cuda-gib 12 --rss-gib 32 --resident-gib 12 `
  --coverage-workspace-mib 96 --warmup-updates 1 --measured-updates 3
```

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 별도 승인된 DEBUG 후보를 비교하고 production 규모를 보존했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 명시적 DEBUG 범위와 전체133관측을 기록했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 동일 physical32 비교, GPU batch tensor 처리와 CPU reader4를 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 검사에서는 OOM 없음, 선택 비용 측정 완료.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 새 graph 가중치의 초기 상태와 unit fixture를 실제 학습 결과와 구분했다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 연결 분리·고립 노드를 숨기지 않았다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
