# HierCP Level 0/1/2 및 실험 파이프라인 검수

검수일: 2026-09-05. 기준 커밋: `056937aaed7a7c2ec5dd2ca6290f015e37eb5ffc`.

## 결론과 범위

**실제 CNN·이종 GAT·랭킹 학습을 구현한 모델이지, 이름만 있는 dummy/toy 모델은 아니다. 그러나 현재 구현은 Level 0/1/2 전체의 정확성과 AGENTS.md 준수를 통과한 상태가 아니다. 아래 결함 수정과 재검증 전에는 fold 1 학습을 시작하지 않는 것을 권고한다.**

현재 저장소의 모델·그래프·학습·ablation·온라인 CP·평가·재사용·설치 경로를 분담 정적 검토했다. 기존 독립 테스트와 추가 NumPy 진단을 실행했다. 이 보고서 이외의 프로젝트 코드와 설정은 수정하지 않았다. 데이터 다운로드, 실제 학습, 의료영상 평가, 체크포인트 로드, Git commit/push는 하지 않았다. 유한한 소스 검토로 모든 잠재 결함이 제거되었다고 보장하지 않는다.

기준 설계는 현재 `AGENTS.md`, `docs/design.md`, `README.md`, `config/train.json`이다. 과거 최초 설계와 현재 설정의 완전한 동일성까지 입증한 것은 아니다. 서버 증거 `C:/Users/lock1/Downloads/resume_evidence.tar.gz`에는 nnU-Net trainer와 로그는 있지만 당시 HierCP GNN 소스·체크포인트는 없다. **아래 현재 코드 결함을 과거 fold 0 결과의 실제 오염이나 Level 2 성능 저하 원인으로 소급 확정하지 않는다.**

## 실행한 검사

| 검사 | 결과 | 의미와 한계 |
| --- | --- | --- |
| Python AST | 46개 파일 통과 | 문법 검사이며 의미·수치 정확성 검증이 아님 |
| JSON | 설정 2개 통과 | 설정 파싱 검사 |
| `python -B -m unittest discover -s tests -p 'test_*.py'` | 64개 중 63개 통과, 1개 skip | 격리된 debug/평가 계약 테스트. Windows symlink 권한 때문에 1개 skip |
| 추가 context sampling 수치 진단 | 1/2/3 hops 결과 동일 재현 | 실제 두 selection 함수의 AST를 그대로 실행. Tensor 변환은 ndarray adapter, SciPy 반경 검색은 정확한 NumPy 전수 거리 검색으로 대체한 독립 debug 검사 |
| 비등방 회전 좌표 진단 | 서로 다른 물리 위치 계산 재현 | 행렬 대수 검사. SciPy affine rasterization이나 의료영상 전체 검사가 아님 |
| 전체 모델 forward/backward·optimizer update | 미실행 | 로컬 Python에 PyTorch/PyG가 없음 |
| 실제 환자 데이터·GPU·전체 학습·전체 평가 | 미실행 | 데이터와 서버 실행 환경이 이 검수에 제공되지 않음 |

로컬 Python은 3.12.14이고 NumPy는 사용 가능하나 PyTorch, torch_geometric, SciPy, nibabel, psutil은 설치되어 있지 않았다. 논리 CPU 수는 16으로 확인했다. Windows CIM CPU/RAM 상세 조회는 접근 거부되어 재시도나 권한 상승을 하지 않았다. GPU 할당·VRAM·실제 서버 RAM·처리량은 미측정이다. 다른 Python 환경의 알려진 위치에서도 이 검사에 사용할 실행 파일을 확인하지 못했다. 의존성을 임의로 설치하지 않았다.

## 모델 규모 및 toy 여부

현재 설정과 실제 생성 코드가 연결된 항목:

- Level 0: 5채널 CT/기하 dense 입력, 48³ 입력 크기, CNN 채널 12 → 24 → 32. ConvBlock 3개와 두 downsample convolution으로 3D convolution 8개가 구성된다. 별도의 가변 크기 native-geometry 그래프를 사용한다.
- 공통 GNN: hidden 128, attention heads 4, relation-specific `GATv2Conv`, residual·LayerNorm·FFN.
- Level 0/1/2 block 수: 3/2/2. 이는 7개 계층 block이지 GAT 한 개씩만 둔 모형이 아니다.
- 각 레벨 node type 수 6/5/3, relation type 수 16/16/5. 설정에 따른 relation-specific GAT 모듈은 정적 계산으로 `3×16 + 2×16 + 2×5 = 90`개이다.
- node-type FFN은 `3×6 + 2×5 + 2×3 = 34`개이며, 이 FFN의 두 Linear 파라미터만 합해 **4,478,208개**이다. 미사용 경로도 포함한 소스 기반 부분 합계로, 전체 모델의 런타임 parameter count는 아니다.
- patient region 24, population prototype 16, cache sample/case 2, 학습 candidate/sample 8, 생성 candidate 128, HierCP 학습 40 epoch가 명시되어 있다. downstream nnU-Net 250 epoch와 별개이다.

따라서 전체를 dummy/toy라고 분류할 근거는 없다. 하지만 큰 parameter 수는 정확성이나 충분한 표현력의 증명이 아니다. 이 규모가 실제 데이터에 충분한지, 원래 최초 연구 요구에 맞는지는 full-data 측정과 기준 모델 비교가 필요하다. 명시된 sample 수·384 context budget·16 prototype 수 자체를 숨겨진 축소로 판정하지 않았다.

현재 파이프라인은 **GNN 랭킹 학습 → score bank/배치 위치 선택 → 별도 nnU-Net 학습 → segmentation 평가**의 두 단계 방식이다. downstream Dice가 GNN으로 역전파되는 end-to-end 시스템은 아니다. 현재 placement-ranking 설계와는 일치하지만 end-to-end로 설명해서는 안 된다.

## 레벨별 판정

| 레벨 | 실제 구현되어 있는 것 | 통과하지 못한 부분 |
| --- | --- | --- |
| L0 Local | full-footprint 검증, native ROI, CNN feature sampling, source/target context, 반경 관계, 두 sampled view, GAT·consistency loss | 추가 hop 무효화, 변형 mask와 mm 그래프의 불일치, surface anchor 의미 대체 |
| L1 Patient | tumor/candidate/region/lesion/liver 그래프와 관계별 message passing, L0 입력, score 및 L2로 전달 | 최종 lesion/liver 갱신 파라미터 미사용, 일부 위치 shortcut 경로의 영향 미검증 |
| L2 Population | train-only prototype fitting, candidate/region/prototype 그래프, 실제 prototype→region 및 최종 점수 경로 | 최종 prototype 갱신 파라미터 미사용, 기여도 보고의 비교 계약 보장 부족 |

**L2 전체가 끊어졌거나 상수 출력을 내는 것은 아니다.** 앞 block과 최종 prototype→region 경로는 살아 있다. 미사용 최종 node 갱신과 Population branch 전체의 유효성은 구분해야 한다.

## 우선 수정해야 할 결함

### F01 / P1 — Level 0의 추가 sampling hop이 실행되지 않음

근거: `hiercp/sample.py:149`, `174`, `294`, `312`.

context node 수가 budget 384를 넘으면 첫 hop의 `_balanced_indices(..., budget=384)`가 이미 384개를 반환한다. 바로 뒤 `selected.size >= budget` 조건으로 loop가 끝난다. 그 결과 `sample_hops=2` 또는 3이 1보다 더 많은 확장을 수행하지 않는다. required/interface 이웃이 budget을 넘을 때 재샘플링하므로, 모든 required 이웃의 보존도 보장하지 않는다.

독립 debug fixture: 768개 context node, budget 384, seed 42. hops 1/2/3 모두 선택 node 384개, 반경 검색 2회(초기 interface 조회 포함), 선택 결과 SHA256 `fe636e589d9cce1e5e0eb9caa38f58006a99ea5bf1708c1516665404ae348cde`로 동일했다. 대상 sample.py SHA256은 `7f1b651ad70837b599324cd5fe06d6c7941c9857d2379027fe2da9a21b77d483`이다.

수정 기준: 요청된 hop 수를 실제 반영하고, sampling과 정확한 hop closure 중 무엇을 보장하는지 계약을 명시한다. node 수·hop 수를 줄이는 해결책을 임의 적용하면 안 된다.

### F02 / P1 — Level 0의 mask 변형과 물리 그래프 변형 불일치

근거: `hiercp/local.py:103`, `109`, `115`, `299`, `313`; `hiercp/sample.py:464`; 도달 경로 `hiercp/curriculum.py:117`.

mask 변형은 voxel 좌표에서 `R @ scale`을 사용하지만 cross-graph 관계는 동일 행렬을 mm 위치에 적용한다. voxel spacing 행렬을 D라고 하면 실제 mask의 물리 변환은 `D R D⁻¹`이어야 하므로, 비등방 spacing에서 축을 섞는 회전이면 둘이 달라진다. curriculum은 실제로 ±90도 회전과 확대/비등방 scale corruption을 생성한다.

독립 대수 fixture에서 spacing (5,1,1), 첫 두 축의 90도 회전, 원점 기준 voxel 점 (1,0,0)을 사용하면 mask 경로는 (0,1,0)mm, 그래프 경로는 (0,5,0)mm로 **4mm 차이**가 난다. 이는 실제 환자에서의 오차 측정이 아니다.

추가 문제: mask 변형의 output shape가 원 footprint bbox로 고정돼 확대·회전 형상이 잘릴 수 있고, 완전히 비면 원 footprint/identity로 되돌아간다(`local.py:119`, `125`). footprint 회전 중심 `(shape-1)/2`와 source anchor `shape//2`(`common.py:495`)를 함께 사용하는 짝수 크기의 정렬도 검사해야 한다. source footprint의 원본 보존 검사와 이 비항등 corruption 경로는 별개이다.

수정 기준: 하나의 물리 좌표·anchor 계약으로 mask, dense feature, canonical topology와 edge attribute를 일치시킨다. 변형으로 필요한 bbox가 커지면 전체 형상을 보존하거나 명시적 오류를 내야 한다. 실제 spacing/shape별 수치 회귀 검사가 필요하다.

### F03 / P1 — L1/L2의 최종 block에 미사용 trainable parameter

근거: `hiercp/model.py:252`, `263`, `738`, `789`, `957`, `961`; `hiercp/pipeline.py:1134`, `1863`, `2039`.

`HeteroGATv2Block`은 이전 x_dict로 모든 관계 message를 먼저 계산하고 새 node 출력을 만든다. 같은 block에서 갱신된 node가 다른 node의 갱신에 다시 사용되는 구조가 아니다.

- Full L1: 마지막 `lesion`, `liver` 출력 전용 norm/FFN과 해당 destination relation conv의 출력이 score 또는 다음 encoder로 전달되지 않는다.
- Full L2: 마지막 `prototype` 출력 전용 norm/FFN과 prototype destination conv가 최종 score로 연결되지 않는다.
- NoPopulation: 마지막 L1 `region` 출력도 소비되지 않는다.

그런데 이 파라미터들은 full/해당 active encoder에서 requires_grad 상태로 optimizer에 포함된다. 현재 최종 완료 gate는 전체 연결을 마지막 epoch에서 확인하므로, 불필요한 학습을 수행한 뒤 완료 검증에 실패할 수 있다. 이는 정적 dependency 분석이며 실제 PyTorch gradient 측정은 아직 하지 못했다.

수정 기준: 원래 message-passing 깊이·표현력을 보존하면서 모든 남겨둔 trainable parameter에 명확한 loss 경로를 설계한다. 단순 dummy auxiliary loss, 임의 zero loss 연결, 검증 우회로 통과시켜서는 안 된다.

### F04 / P1 — 오래된 derived score bank의 출처를 새 checkpoint로 재표기

근거: `tools/downstream_level_ablation.py:755`, `783`, `619`, `628`.

두 mode의 NPZ가 존재하면 현재 model/source와의 결합을 확인하지 않고 score를 재사용한다. 이후 index에는 현재 checkpoint/source bank의 SHA를 기록한다. checkpoint나 source가 바뀐 경우 이전 score와 새 provenance가 혼합될 수 있다.

수정 기준: 재사용 전에 source bytes·candidate 순서·checkpoint·prototype·설정·score 생성 코드의 계약을 확인하고, 불일치는 명시적으로 거부한다. 파일 존재만으로 인증하지 않는다.

### F05 / P1 — 부분 bank 복구 시 overwrite 승인 없이 기존 결과 교체

근거: `tools/downstream_level_ablation.py:755`, `912`, `931`.

두 NPZ 중 하나만 남아 있으면 all-exists 재사용 분기를 타지 않고 두 mode를 모두 계산한다. 뒤의 `os.replace`가 이미 존재하는 한쪽 결과도 `--overwrite-banks` 없이 교체한다.

수정 기준: 전체 작업의 충돌·부분 상태를 쓰기 전에 확인한다. 기존 파일은 계약이 맞을 때 재사용하거나 새 출력 위치를 사용하며, 덮어쓰기는 명시적 승인 범위에서만 수행한다.

### F06 / P1 — nnU-Net 완료 결과 재사용에 학습 입력 계약 대조 누락

근거: `tools/downstream_level_ablation.py:1029`, `tools/nnunet.py:480`.

final checkpoint 및 validation 파일 존재만으로 완료 재사용을 결정한다. 현재 bank/trainer/config/plans/split과 저장된 학습 결과의 결합을 검증하지 않는다. 따라서 bank를 제대로 새로 계산했어도 이전 nnU-Net 결과를 그대로 붙일 수 있다. 일반 online benchmark의 training-contract 보호가 이 entrypoint들에 동일하게 적용된 상태는 아니다.

수정 기준: 완료 결과 및 resume 상태를 immutable training-input contract에 결합하고, provenance 없는 기존 결과를 자동 인증하지 않는다.

### F07 / P1 — GNN ablation 기여도 보고가 동일 실험 조건을 검증하지 않음

근거: `tools/ablation.py:146`, `175`, `237`, `283`, `357`.

Full 기준은 파일 존재/mode 중심으로 검사한다. ablation에는 epochs/seed override가 가능하지만 보고 시 서로의 cache/split/prototype/config/seed 계약을 대조하지 않는다. partial checkpoint도 metric이 유한하면 contribution 계산에 들어간다. 보고서는 같은 cache/split/prototype/seed/curriculum이라는 문장을 항상 출력한다.

수정 기준: 비교에 필요한 동일 조건과 training_complete를 실제로 확인한다. 미완료·비교 불가 상태를 정식 기여도와 분리한다. 이것은 **현재 GNN 보고기의 문제**이며 과거 downstream 표의 조건이 실제 달랐다는 증거는 아니다.

### F08 / P2 — 실제 간 표면이 없으면 내부 조직으로 의미 대체

근거: `hiercp/spatial.py:512`, `555`.

ROI에 `liver_depth<=boundary_depth` band가 없으면 가장 얕은 내부 조직 주변을 `liver_surface`로 지정한다. 검증은 context와의 비중복 등을 보지만 이 anchor가 실제 표면 깊이 조건을 만족하는지는 보장하지 않는다.

수정 기준: 실제 표면을 찾도록 ROI/anchor 정책을 명시적으로 설계하거나 필수 표면 부재를 오류/명시적 별도 의미로 처리한다. 조용히 다른 조직으로 대체하면 안 된다. 실제 데이터 발생 빈도는 미측정이다.

### F09 / 검증 공백 — 기존 smoke가 마지막 block의 미사용 파라미터를 놓침

근거: `tools/smoke.py:513`, `550`, `tools/ablation.py:394`; 실행 gate는 `hiercp/pipeline.py:2039`.

기존 smoke는 각 레벨 첫 block에서 일부 파라미터에 gradient가 있으면 통과한다. 모듈 단위 any-gradient 검사도 전체 parameter 연결을 보장하지 않는다. 따라서 과거 smoke 통과와 F03은 모순되지 않는다.

수정 기준: 모든 active block/관계의 loss dependency를 확인하고, finite gradient·실제 optimizer update 및 ablation별 활성 경로를 분리 검증한다. 정상적인 conditional/empty-node 경로도 명시적으로 다룬다. 실패를 최종 epoch까지 미루지 않는다.

## 부차적 위험과 미검증 영역

- **잠재 shortcut:** 절대 좌표와 일부 tumor spatial edge 열은 마스킹하지만, source tumor의 실제 region topology와 `same_region` 열은 남는다(`hierarchy.py:328`, `444`, `model.py:110`). positive가 source 위치인 curriculum에서 일부 난이도 구분 단서가 될 수 있다. 동일 region/prototype의 corrupted negative도 있으므로 완전한 label leak이나 성능 영향이 입증된 것은 아니다. 같은-region 대조 실험 등으로 확인해야 한다.
- **자원 활용:** downstream rescoring은 source 하나를 `collate_samples([sample])`로 처리한다(`downstream_level_ablation.py:823`). sample 내부 후보는 chunked/batched이므로 후보 모두가 직렬이라는 뜻은 아니다. loss/metric도 batch 내 sample별 Python loop가 남아 있다(`loss.py:126`, `238`). 이 경로의 batch 후보별 VRAM/처리량 근거가 없으며, 실제 병목 크기는 미측정이다.
- **GNN multi-GPU:** 현재 production은 여러 CUDA device가 보이면 명시적으로 거부한다. 숨겨서 한 GPU만 쓰는 동작은 아니지만, 검증된 multi-GPU 구현은 없다는 제한이다. 서버 할당을 확인하지 않고 활용 규칙 준수를 인증할 수 없다.
- **통계 진단:** 보조 Wilcoxon의 ValueError를 사유 없는 None으로 바꾸는 경로가 남는다(`online_eval_v2.py:707`). 현재 표의 주된 permutation p-value가 틀렸다는 발견도, 0점 대체 발견도 아니다.
- **서버 원본 resume:** 첨부 `run/run_training.py:68–78`은 `--c`에서 checkpoint가 전혀 없으면 경고 후 신규 학습으로 넘어간다. 필수 상태 누락을 승인 없이 다른 실행으로 대체하는 경로이다. 제공된 epoch 로그에서 이 분기가 실행됐다는 증거는 없다.
- **로그 쓰기 실패:** 서버 base trainer가 파일 기록 실패를 콘솔에 보고하고 계속하는 경로가 있다. AGENTS는 콘솔 출력도 허용하므로, 이것만으로 필수 기록 규칙 위반이라고 확정하지 않았다. 저장 로그 내구성 위험으로 구분한다.
- **버전:** 현재 HierCP GNN과 과거 서버 GNN의 동일성, 실제 체크포인트별 optimizer 상태, 어떤 checkpoint가 기존 prediction을 생성했는지는 미검증이다.

## 확인된 보호 및 정상 경로

- 원본 source footprint voxel 수를 검사하고, 48³ dense view와 가변 native geometry를 구분한다. canonical node/edge 및 ROI 상한은 조용한 truncation 대신 명시적 오류이다. 단, F01/F02의 sampled/corrupted 경로는 별도 결함이다.
- PyG variable-node collation과 candidate 순서 검사가 실제 연결된다. 전체 GNN forward를 sample마다 별도로 실행하는 구조는 아니다.
- region fitting sample 이후 전체 organ voxel을 할당한다. descriptor CT에서는 종양을 제거한 주변값을 사용한다.
- prototype fitting caller는 train split만 전달하며, cache 준비는 train/validation 교집합과 bank training ID 불일치를 거부한다. 확인한 정상 경로에서 validation prototype-fit 누수는 발견하지 못했다.
- ablation은 제외 encoder를 동결하고 활성 feature 폭에 맞는 score head를 구성한다. 구조적 제거 자체와 F07의 비교 계약 검증은 별개이다.
- production/debug/ablation 분리, 주요 환경 검사, 자식 명령 실패 전파가 존재한다.
- 평가 재실행은 새 출력 경로·cohort·입력 hash·completion marker를 확인한다. 이는 학습/bank provenance의 F04–F07을 대신하지 않는다.
- custom trainer installer는 명시적 overwrite·백업·복사 후 hash 확인·실패 rollback 경로가 있다.

## 후속 검증 순서

1. 과거 fold 0 원본 결과·로그·bank·checkpoint를 보존한다. 이번 보고서로 원본을 폐기하거나 resume 상태를 verified로 바꾸지 않는다.
2. F01/F02/F03의 모델·기하 경로와 F04–F07의 재사용/비교 계약을 먼저 수정하고, 별도 debug 재현 테스트를 추가한다. 모델 깊이·해상도·candidate·데이터 규모를 임의 축소하지 않는다.
3. 서버 실제 GNN 소스/config/checkpoint provenance와 현재 코드의 차이를 확인한다. 변경된 구현으로 얻은 수치를 기존 fold 0과 동일 버전 결과처럼 묶지 않는다.
4. 호환 환경에서 full 설정 모델의 parameter census, 단계별 gradient/optimizer update, batch-vs-single 및 chunked-vs-full 동치, 비등방·짝수 bbox·확대·회전 기하 검사를 수행한다. synthetic 검사는 debug로 명시한다.
5. 실제 서버 할당에서 batch/worker/VRAM/처리량과 full-data preflight를 확인한 다음, 사용자가 정한 **fold 1 추가 실험** 범위만 진행한다. 이 검수는 남은 모든 fold 실행을 승인하거나 시작한 것이 아니다.

현재 fold 0에서 Level 2 제거 시 일부 지표가 더 좋았다는 관찰은 유지된다. 그러나 그 관찰을 Level 2 설계 자체의 불필요성으로 확정할 수는 없고, 현재 코드 결함이 그 관찰의 원인이었다고도 단정할 수 없다.

## 작업 완료 체크리스트

아래는 이번 검수 작업과 현재 검증 상태를 구분한 체크리스트이다. 미확인·미충족 항목은 체크하지 않았다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 다만 서버 실측은 미수행이다.
- [ ] GPU, CPU, RAM 활용 상태를 모두 측정하거나 확인했다. 로컬 논리 CPU만 확인했고 상세/서버 자원은 미확인이다.
- [x] OOM 회피를 위해 모델을 축소하지 않았다. 이번 검수에서 모델 학습/OOM은 발생하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다. 독립 진단은 명시적 synthetic debug이며 설정 파일은 불변이다.
- [x] dummy, placeholder, random fallback을 실제 연구 결과로 사용하지 않았다.
- [ ] 핵심 모듈이 모두 forward, loss, gradient와 optimizer에 연결되어 있다. F03 미충족, 실제 GPU gradient 검사도 미실행이다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다. 검수 보고서만 새로 추가했다.
- [x] 독립 테스트와 전체 모델 smoke, 전체 학습 및 전체 평가를 구분해서 보고했다.
