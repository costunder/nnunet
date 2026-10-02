# v2.2 L0 — 실제 CT의 CNN 특징 기반 희소 그래프 DEBUG 구현

2026-10-02. 구현과 실제 3D 좌표·연결 시각화를 위한 별도 연구 후보다. Production 기본 모델, 학습 설정, 이전 checkpoint와 결과는 변경하지 않았다. 이전 mean L0의 exact resume나 학습 완료 모델이 아니다.

## 구현한 경로

```
원본 donor / recipient의 간 내부 국소 CT crop
→ 기존 OrganPyramid CNN [12,24,32], convolution [2,3,3], stride 1/2/4
→ 간 내부 fine 위치의 세 scale CNN 특징 68D
→ 후보 기준 near / mid / wide 각각 CNN 특징·공간 분산으로 16위치 선택
→ 원래 anchor의 추상 query 1개 + 공간 context 48개
→ 공간 3NN / 특징 1NN의 대칭 합집합으로 작은 그래프
→ 3층·128D mean GraphSAGE, 층별 자기 정보 residual 1회
→ query / near / mid / wide를 분리한 readout → scene 128D
→ 기존 donor–recipient fusion → L0 128D
→ 기존 L1 / L2 / 전체 objective → backward → fresh AdamW update
```

CNN은 **unique crop당 한 번** 실행하고 모든 선택 위치가 세 feature map을 공유한다. 환자 전체를 한 번 인코딩하는 구현은 아니다. 여러 scene의 선택·연결·message passing은 batch tensor로 처리한다. 큰 fine graph의 edge 생성·GAT·EZ-SP 병합은 이 새 L0에 들어가지 않는다. 작은 adjacency와 batched matrix multiplication을 사용하며, 노드별 CNN 재실행이나 그래프별 개별 backward를 하지 않는다.

노드 입력은 CNN 특징과 원래 후보 기준 상대 좌표다. CT 평균·곡률·분산 등의 handcrafted 조직 특징을 추가하지 않았다. 아래 분산은 **노드 선택 거리의 단위 정규화**에만 쓰고 모델 입력에 넣지 않는다.

## 이번 DEBUG profile과 선택 규칙

이번 실제 검사는 기존 margin 10mm native crop에서 수행했다. Query 특징은 원래 anchor 반경 3mm 안의 유효 CNN 위치에서 읽고, near는 5mm 이하, mid는 5–10mm, wide는 10mm 밖의 기존 crop 내부다. Wide의 범위는 기존 crop 경계로 결정하며 간 전체로 확장하지 않는다.

각 band에서 전체 유효 CNN 위치를 검토하고, 첫 위치는 anchor에 가장 가까운 곳으로 선택한다. 이후 후보는 특징과 공간의 혼합 거리에서 기존 선택점들과 멀리 떨어진 위치를 순차 선택한다. 각 항은 같은 scene/band에서 측정한 `2 × mean(||v − mean(v)||²)`로 나누므로 CNN 특징 거리와 mm 공간 거리의 척도를 맞춘다. 두 항의 가중치는 각각 1이다. 의미 있는 조직 분류·CP 중요도를 학습한 selector라고 주장하지 않는다.

첫 시각화에서는 CNN 거리보다 공간 거리의 척도가 커서 외곽 선택이 박스 모서리에 치우쳤다. 초기 결과를 보존하고 척도 정규화를 수정했다. 동일 좌표에서 CNN 특징만 바꾸면 선택점이 달라지는 GPU 검사를 추가했다. 공간 단위를 100배 바꿔도 정규화 후 선택은 같음을 검사했다. 특징 분산이 0이거나 유한하지 않은 band를 geometry-only 선택으로 조용히 대체하지 않는다.

16개/band, 3+1 이웃, 3층/128D는 이번에 명시한 **별도 DEBUG 비교 구조**다. Production의 숨겨진 노드 cap이나 기존 그래프 축소 설정이 아니다. Band에 16개보다 적으면 모두 보존하고 padding은 실제 노드/edge로 집계하지 않는다. Query/band의 필수 coverage가 없으면 오류로 보고한다. 간 밖 CT나 잘못된 입력으로 채우지 않는다.

Hard selection의 index에는 일반적인 gradient가 흐르지 않는다. 선택된 위치의 **현재 CNN 특징을 다시 gather**하여 CNN·node projection·SAGE·readout·fusion까지 loss gradient를 전달한다. 좌표를 학습하는 deformable exploration, 뿌리 성장, CP saliency 모델이 아니다.

## 간·좌표·주석·관계의 의미

- 공간 노드는 실제 native organ mask 내부 위치다. 원래 anchor, crop origin, spacing 및 stride로 좌표를 계산하며 padded tensor의 midpoint로 anchor를 바꾸지 않는다.
- Query는 원래 anchor에 표시하는 추상 token이다. 오목한 종양 component의 bbox midpoint가 간 밖이면 이를 CT 공간 노드라고 오해하지 않도록 구분한다. 원래 anchor를 이동하지 않는다.
- CNN 입력과 feature interpolation은 간 밖 CT를 차단하고 유효 organ corner만 사용한다. 간 밖/padding 값을 NaN으로 바꾸어도 결과와 graph가 바뀌지 않는 GPU 검사를 통과했다.
- 세 scale에서 지원되는 위치만 context 선택에 사용한다. 이번 266 scene 중 fine organ 위치의 최대 **14.2231%**는 coarse-scale corner 지원이 없어 제외됐다. 모두 보존하는 무손실 압축이라고 표현하지 않는다. Pool/coverage 수는 raw report에 기록한다.
- P/U와 `lab==2`를 selector에 입력하지 않는다. 종양 주석은 원본 CT annotation의 **별도 시각화 overlay**다. 그래프 노드가 종양 mask라는 뜻이 아니다.
- Edge는 공간/특징 이웃 간 message relation이다. 혈관·voxel 경로를 추적한 선이 아니다. 직선이 간 밖을 가로질러 보이더라도 그 선을 따라 외부 CT를 읽지는 않는다.
- 연결되지 않은 경우의 MST 연결은 표시·집계한다. 이번 266 scene은 처음부터 연결되어 새 MST edge가 0개였다.

## 유지한 정답과 학습 경계

P는 실제 관측된 적격 종양 anchor, U는 원래 128개 미관측 comparison center다. U를 CP 부적합, P를 특정 donor에 맞는 CP 정답으로 재정의하지 않았다. 동일 case/donor와 원본 assignment 결속을 유지했다. Basic CP 80%, seed42, split, 원본 mask, L1/L2의 architecture와 equations, 기존 ranking/observation/alignment loss와 정규화 계약은 변경하지 않았다.

새 L0로 저장된 DEBUG cohort의 eligible support 6개를 모두 다시 인코딩하고 teacher/cluster plan을 다시 맞췄다. 이전 mean L0의 memory를 섞지 않았다. **이 support는 production 전체 support가 아니다.** 복제 update에서 L1/L2 가중치도 실제 갱신되지만 구조·연산식은 그대로다.

CNN과 기존 pair fusion의 가중치는 무결성을 확인한 로컬 DEBUG saved step 4에서 복사했다. Server epoch22/39 학습 가중치가 아니다. Node projection·GraphSAGE·scene readout은 seed42의 새 초기값이다. 이로 만든 시각화를 암 관련 특징의 학습 완료 결과로 제시하지 않는다.

## 실제 검사 결과

RTX 5070 Ti 16GiB, FP32, autocast/TF32 off, deterministic on, worker4, CUDA budget12GiB/RSS32GiB/resident12GiB의 명시적 DEBUG 실행이다.

| 검사 | 실제 범위와 결과 |
| --- | --- |
| GPU 단위 검사 | 9개 PASS. 선택의 feature 의존성·RNG·native geometry·coverage·padding·MST·모든 parameter gradient 검사 |
| 실제 CT 전체 case | liver_66 원래 P5+U128=133개, 32/32/32/32/5로 처리. Donor/recipient 총266 scene |
| 그래프 크기 | 모든 scene 49개: query1, near16, mid16, wide16. Pair당98개 |
| 표시한 실제 연결 | Donor1+P5+U3의9 scene에서 undirected127–152개. 이는 전체266 scene의 edge 범위가 아님 |
| 실제 loss/update | Physical/effective32, accumulation1, P×U ranking135쌍, 전체 objective1회, fresh AdamW |
| Gradient와 parameter 갱신 | CNN·node projection·SAGE·graph readout·fusion·L1·L2 모두 finite positive gradient. CNN/SAGE/L1/L2 실제 delta>0 |
| 모델 규모 | 전체/trainable1,242,198 parameter |
| 메모리 | Peak allocated2.134GiB, peak reserved2.662GiB, RSS5.759GiB |
| DEBUG timing | 전체133 forward 합1.348초; update forward0.401/backward0.120/optimizer0.053초. Support/teacher/CPU3D export 포함35.137초 |
| 3D 화면 검사 | 10개 PASS, browser errors0. P/U 전환, 회전, native-mm node detail, 전체 간/국소 보기, 종양 overlay, mobile 확인 |
| 원본 보존 | Production121파일, 이전 checkpoint/assignment, 작업 전 dirty3파일 SHA256 보존 |

위 시간은 한 case와 DEBUG support의 짧은 검사다. 데이터 로딩·실제 production checkpoint 저장·전체 support refresh·전체 validation을 포함한 epoch 비교가 아니다. **3시간/epoch 해결이나 v1 이상의 정확도를 입증하지 않는다.** 장기 학습, checkpoint/ready 작성, production 기본 교체는 수행하지 않았다.

최종 raw report와 GPU/화면 검사 기록은 `validation/l0_sparse_feature_DEBUG_20261002/`에 보존한다. 초기 척도 문제 결과는 기존 `work/sparse_feature_CT_DEBUG_20261002/`, 최종 결과는 별도 `work/sparse_feature_balanced_CT_DEBUG_20261002/`에 있다. Raw CT, mask, 가중치와 3D anatomy payload는 Git 공개에 포함하지 않는다.

## 로컬 재현 명령

기존 입력 경로가 있는 로컬 workspace에서만 실행하는 별도 DEBUG 명령이다. Output과 visual은 존재하지 않는 새 경로를 사용한다. 전체 학습 명령이 아니다.

```powershell
& .venv/Scripts/python.exe tools/verify_sparse_feature_ct_debug.py `
  --run work/local_cnn_experiment_resume_DEBUG_20261001/resumed `
  --assignment work/v222_v1_full_training_20260924/cache/pair_assignment.json `
  --assignment-receipt validation/reference_finite_shadow_20261002/actual_CT_DEBUG_report.json `
  --case liver_66 `
  --output work/sparse_feature_replay_DEBUG_20261002 `
  --visual work/sparse-feature-replay-debug.html `
  --nodes-per-band 16 --query-radius-mm 3 --near-radius-mm 5 --mid-radius-mm 10 `
  --workers 4 --cuda-gib 12 --rss-gib 32 --resident-gib 12
```

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 초기 결과도 보존했다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다. CNN·L1/L2 유지, 별도3층/128D 연구 후보 명시.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 새 sparse L0 비교 구조를 분리하고 원래 P5/U128 case 전체를 검사했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. DEBUG context139/support6/표시9 scene과 명시적16개/band를 구분했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 원래32 유지, batched selection/message passing.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실제 GPU peak/RSS/CPU16/worker4 기록.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 최종 검사 OOM 없음; production 속도 해결 주장 없음.
- [x] 디버그 설정과 최종 설정을 분리했다. Production profile은 변경하지 않았다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 실제 CT·검증된 CNN snapshot·실제 forward 좌표 사용; 새 graph 초기값 출처 명시.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제32pair 전체 objective 검사.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. Coverage와 초기 metric 문제도 기록했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 장기 학습/전체 평가/정확도 개선은 미검증이다.
