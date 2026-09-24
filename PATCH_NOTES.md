## v2.22 r6 — tqdm 진행 화면과 실행 중 재접속 (2026-09-24)

- `run_v222_server.py`를 터미널에서 직접 실행하면 작업은 분리된 프로세스로 유지하고 화면에 두 줄 tqdm을 갱신한다. 상세 출력은 run의 `console.log`에 저장한다. 화면의 Ctrl+C는 worker를 중단하지 않는다.
- 독립 `tools/watch_v222_server.py`는 기존 실행의 로그/완료 기록을 읽어 재시작 없이 연결한다. CT/graph/support/epoch query 완료 수, optimizer step/loss를 실제 이벤트로 표시한다. 미제공 진행률은 만들지 않으며 재접속 직후 과거 로그 재생 속도로 ETA를 계산하지 않는다. 실패는 완료로 표시하지 않고 마지막 로그를 보여준다.
- UI 프로토콜과 실제 분리 프로세스 완료/화면 중단 후 지속, 기존 runner/중단 보호 검사 총9개 통과. 모델·전체 데이터·학습 설정 불변. 원격 MobaXterm 화면 실검증/전체 학습 수행 주장이 아니다. `SERVER_V222.md`에 기존 실행 연결 및 새 실행 명령을 갱신했다.

## v2.22 r6 — 원본 CT 준비의 중복 벤치마크 제거 (2026-09-24)

- 서버 로그는 학습이 아니라 `observations`의 128개 CT 워밍업 뒤 동일 128개를 worker1로 다시 읽는 단계였다. 이 경로는 이후 여러 worker 후보에서도 같은 CT를 반복 측정했다.
- 서버 도구를 기존 v1의 실제 작업 보존형 스케줄러에 연결했다. 큰 CT부터 한 번씩 처리하고 완료 결과를 즉시 저장한다. 관측 규칙·모델·전체 데이터·seed·학습 설정은 불변이다. 개별 시작/완료와 10초 heartbeat를 출력한다.
- 로컬 전체131개를 정확히131회 처리, 관측14,102개·donor527개 및 전체 중심/정답/split/donor 배정 동일. 실제 작업 wave 합계361.73초이며 서버 소요시간 예측이 아니다. 관련7검사 통과(0skip). 증거: `validation/v222_r6/observation_single_scan.json`.
- `tools/stop_v222_observation_job.py`는 기록된 해당 observations 자식의 소유자/PID/명령/부모/진행 단계를 검증한다. `--stop` 명시 시 그 자식만 중단하고 wrapper의 실패 기록을 확인한다. 다른 단계로 진행됐으면 거절한다. 보호 조건 단위검사는 통과했으나 Linux 실제 신호 실행은 로컬 Windows에서 검증하지 않았다.
- paired 그래프 준비의 별도 benchmark 및 전체 학습 성능을 이번 수정으로 검증했다고 주장하지 않는다. 서버 재시작은 사용자가 명령을 실행해야 한다. 기존 출력은 보존하고 새 출력 경로를 사용한다.

## v2.22 r6 — 서버 실행용 Git 배포 준비 (2026-09-24)

- **사용자 서버 출력으로 확인한 최신 상태:** `ece-agpu16`, checkout `/home/aicompetition06/Medical/HierCP-v222-r6`, 원본 `Medical/Data/image`·`Data/labels` 존재, Python3.10.18의 `nnunet` 환경 활성화. 사용자 할당은 물리GPU6. 지정한 UUID `MIG-774a3cc0-0169-5e18-b5d9-fe1b7a1d6ce7`에서 PyTorch가 A100-SXM4-80GB MIG1g.10gb/9.5GiB/1device 인식한 출력을 받았다. 이를 다시 묻거나 GPU0/과거GPU5/A6000으로 취급하지 않는다. 긴 Python heredoc 대신 새 `tools/run_v222_server.py` 실행 명령을 제공한다. 현재 환경을 유지하고 원본 관측→실제graph DEBUG→전체paired cache→batch32/9GB DEBUG→auto-batch GNN40epoch를 순차 실행한다. 단계 실패 시 후속 실행 차단·기존 결과 보존. 실행기 제어 단위3검사 통과; 모델소스 hash 불변. 서버 본학습 시작 증거는 아직 받지 못했다.

- **MIG10GB 사전 메모리 smoke 실측:** RTX5070Ti에 PyTorch allocator9GB 상한을 강제하고 초과 할당 OOM을 확인했다. 실제 큰 그래프6배치씩과 저장된 실제 support1,216개, 전체5,550,806 parameter 사용. physical16 최대allocated3.695GB/reserved3.813GB, physical32 6.404GB/6.717GB로 forward/backward/모든 gradient/모듈 update/저장·재개 비트일치/eval 통과. physical64는9GB 상한에서 OOM. `tools/smoke_v1_memory_limit.py`, `validation/v222_r6/memory_limit_9GB_DEBUG.json`, `docs/v222_mig10gb_smoke_20260924.md`에 기록했다. 모델/그래프/production설정은 불변. 전체11,279 support/전체epoch/A100 MIG 속도는 검증하지 않았고 CUDA context 등 allocator 밖 메모리는 상한에 포함되지 않는다. profile 도구에 `--batch-size`, `--allocator-gb` 추가;10GB에서는 기존batch64 명령을 쓰지 않는다.

- **코드 전용 배포 검증 완료:** 새 Git checkout(autocrlf=false)에서 소스 해시/전체5,550,806 parameters 동일. 원본131개에서14,102관측·527donor를 재생성해 이전 중심/정답/donor 배정 전부 일치. 회귀20+재생성2검사 통과(0skip), 실제 원본→6개 full graph→L0/L1/L2 gradient/update 및 checkpoint 재개·epoch 평가 DEBUG 통과. 상세 `validation/v222_r6/code_only_release_checks.json`. 로컬 전체 그래프 재생성/40epoch/서버 학습/nnU-Net 평가를 완료했다고 주장하지 않는다. 캐시 전송 없이 서버에서 생성한다.

- 실행 진입점은 `run_v222_v1_l0.py`. 사용자 지시에 따라 **코드만 Git 이동, 서버 원본 CT로 재생성**한다. 캐시 전송/경로 연결 절차는 폐기했다. `SERVER_V222.md`에 명령을 기록했다.
- `tools/v1_server.py observations`는 기존 raw 관측 생성 함수와 고정 `config/split_cp80_fold0.json`을 사용한다. 서버에서 관측→paired 그래프→학습 순서로 생성하며 로컬 CT/cache/checkpoint/fixture는 전송하지 않는다.
- Windows/Linux checkout 줄바꿈으로 고정 소스 SHA가 달라지지 않도록 `.gitattributes`를 추가했다. 고정 모델 파일 내용과 원본 manifest는 변경하지 않았다.
- 테스트 입력은 `HIERCP_TEST_OBSERVATION_INDEX`, `HIERCP_TEST_FIXTURE`로 지정할 수 있다. 실제 metadata가 없는 checkout의 해당 integration 검사는 명시적으로 skip하며 전체 통과로 세지 않는다. 로컬20검사 통과; GPU 실제 CT 검증 요약은 `validation/v222_r6/`.
- 로컬 전체 학습은 사용자 지시에 따라 저장·중단 상태. 서버 학습/서버 GPU runtime 검증은 아직 미실행이다.

## v2.22 r6 — 중간 저장·재개와 지속 실행 성능 수정 (2026-09-24)

- 원래 epoch 끝에만 저장해서 중단 시 해당 epoch가 사라지던 실행기를 수정. 매 optimizer batch 및 support encoding batch 이후 가중치/optimizer/RNG/진행 위치/고정 L2 cluster plan을 원자적으로 저장한다. 재개는 새 run에 기록하며 기존 결과를 보존한다.
- GPU 전송 직전 graph tensor clone을 제거해 pinned host memory를 보존한다. PyG store만 복사하므로 원본 CPU batch는 유지된다. 모델/그래프/seed/physical64/전체 코호트 계약은 그대로다.
- 실제 가변 graph batch의 CPU wait/H2D/forward/backward/optimizer/allocated/reserved를 기록한다. 고정 batch 반복 calibration만으로 지속 성능을 보장하지 않는다. 사용하지 않는 CUDA allocator cache 반환 효과는 실제 비교 후 기록한다.
- 기존 실행은 49 optimizer step 후 중단했고 epoch checkpoint가 없었다. 전체14,102개 준비 graph는 재사용한다. DEBUG와 전체 학습은 구분하며, 재개 수치 동일성 및 개선된 실행 상태는 검증 결과를 확인한다.
- 실제 CT 연속 실행과 재개의 loss/모든 parameter/optimizer/support memory 비트 동일성 통과. 전체 실행기 중단→재개→epoch 갱신→평가→최종 저장 smoke 통과. 같은64개·46,852,381edge batch277.81→48.00초, reserved21.39→12.77GB; 10batch 합계759.14→421.04초. 모델·그래프·loss는 동일. 전체 epoch 개선율이나 학습 성능으로 일반화하지 않는다. 상세: `docs/v222_v1_execution_r6_20260924.md`.

## v2.22 r5 — 실제 빈 recipient context 처리 (2026-09-24)

- **실행 확인 17:19:** 전체 graph14,102개 준비 및 train11,279개 support 인코딩 후, 전체40epoch 실행에서 첫 실제 optimizer step 완료. loss1.8510187864, physical/effective batch64, loader8, gradient 전부 존재·유한. 시작 증거는 `work/v222_v1_recovered2_training_20260924/training/first_optimizer_step.json`; 전체 학습/nnU-Net 완료 주장이 아니다.
- 깊이 4mm 초과의 target_context가 실제로 없는 두 위치에서 입력 검사가 준비를 중단하던 오류를 수정했다. 노드를 만들거나 조건을 완화하지 않고 기존 empty-shell 처리와 실제 간 표면 CT 노드를 연결한다. source 및 다른 node type의 필수 검사는 유지한다.
- 실제 실패 CT 두 위치의 전체 모델 역전파·parameter update와 수신 CT 민감도를 확인했다. 비어 있지 않은 기존 그래프의 CT·node·edge·epoch sampling은 동일하다. 검증한 정확한 source hash 쌍으로만 기존 그래프를 재사용한다.
- 두 번째 준비는 4,800개 그래프에서 실패했으며 optimizer 0회였다. 전체 학습과 DEBUG 검증을 구분한다. 후속 실행: `work/v222_v1_recovered2_training_20260924/`.

## v2.22 r5 — 수신 해상도에 표현 불가능한 donor draw 및 준비 복구 (2026-09-24)

- 첫 전체 준비가5복셀 donor의 nearest-neighbor 소실로 중단됐다. 실제 optimizer 실행0회. 전체 해상도 조합 사전검사를 추가해 같은 오류3건을 사전에 확인한다.
- GNN pair의 해당 draw만 같은 training pool에서 seed 기반으로 재배정한다. 관측14,102개와 사용 donor527개, 원본 mask·보간·모델·CP 조건은 유지한다. 실제 실패CT3건 그래프 생성 회귀 검증 통과.
- 기존 그래프를 무결성 검사 후 새 실행에 hard link로 재사용한다. 공유볼륨과 graph worker의 RAM 비용을 분리해 병렬화 후보를 실측한다. 감독 프로세스가 준비/optimizer 시작/완료/오류를 별도 상태로 기록한다.

## v2.22 r5 — paired 전체 코호트 학습 실행기 (2026-09-24)

- 사용자 학습 지시에 따라 전체 14,102 관측 위치의 paired 준비와 40-epoch L0/L1/L2 학습 실행기를 추가했다. 원본 CT 복제 없이 donor 공통 저장, 압축 canonical 그래프, 전체 epoch coverage 검사를 사용한다.
- strict deterministic CUDA 3D grid sampling backward 오류를 같은 trilinear gather로 수정했다. seed/결정론 설정과 모델/그래프 규모를 유지한다.
- CPU worker·GPU physical batch 실측, persistent thread loader·공유 RAM cache·graph bucket·disjoint batch를 연결했다. 이 항목은 구현 기록이며 학습 완료 주장이 아니다. [실행·검증 범위](docs/v222_v1_training_20260924.md).
- 새 native bank/nnU-Net 온라인 연결은 여전히 미완료다. 기존 r4 실행기와 캐시를 새 결과로 재표기하지 않는다.

## v2.22 — v1 방식 L0 적용 (2026-09-24, 별도 r5 구성)

- 사용자 요청으로 CT-only v1 문맥 그래프 L0를 기존 L1/L2에 연결. L0 4,718,420 / 전체 5,550,806 parameters. 수작업 특징·종양 내부 node·중심 가림·PPR/A*·9천만 CNN은 새 경로에 넣지 않았다.
- `v1_local.py`의 실제 paired 그래프 입력, query가 donor로 등장하는 support까지 제외하는 누수 방지, 기존 observation loss, 실제128 후보 점수 경로 구현. 단위6개와 실제 CT gradient/optimizer/후보 점수 DEBUG 통과.
- 실행기 `run_v222_v1_l0.py`; [상세와 남은 범위](docs/v222_v1_l0_20260924.md). 전체 학습/평가 및 기존 native online bank 전환은 미완료. 새 구조의 성능 개선을 주장하지 않는다. 이전 코드/설정은 `versions/v2.22/before_v1_l0_apply_20260924/`에 보존했다.

## 2026-09-24 이전 정정 — 독립 CNN 기준선 해석 철회 (문서만 수정)

- 90,337,920-parameter 독립 CNN은 승인된 L0 기준선이 아니며 실제 L1/L2/CP 경로에 연결해 학습한 결과도 아니다. 기존 실행 자료는 미학습 진단 증거로 보존한다.
- 뿌리 성장형은 구현 지시가 아니라는 사용자 답변을 반영한다. 새 다중 스케일 CNN 제안도 채택/구현하지 않았다.
- 보존 v1 L0는 원본 SHA 대조 및 구조 계산 기준 5,600,740 parameters(CNN 120,804 포함). 과거 서버 가중치와의 정확한 revision 일치는 아직 확인되지 않았다.
- 이번에는 모델/설정/입력/가중치를 변경하거나 학습을 시작하지 않았다. 코드 수정 완료로 읽지 않는다. 상세는 `gpt_handoff.md` 최상단을 따른다.

## 2026-09-24 이전 정정 — 요청한 그래프와 CNN-only 산출물의 불일치

- 사용자는 CNN 특징맵 시각화를 반려하고 그래프 요청임을 정정했다. `v22_cnn_l0_20260924`의 실제 CT 실행은 보존하되 요청한 L0 그래프 완료로 취급하지 않는다. 이 코드에는 노드/엣지/성장 경로가 없으며, v2.2 최종 그래프 설계가 승인됐다는 근거도 아니다.
- 성장형 local graph와 다른 CNN 특징 기반 graph 중 의도한 범위를 확인 중이다. 기존 파일을 삭제하거나 임의의 고정 격자 그래프로 대체하지 않았다.

## 2026-09-24 이전 — CNN-only로 해석해 구현한 기준선 (`v22_cnn_l0_20260924`)

- 사용자 지정 명칭 v2.2로 CNN L0를 분리했다. 과거 v2.2 및 v2.21/v2.22 그래프 코드·결과 보존. 새 실행기는 `tools/run_v22_cnn_l0_one_case.py`, 기존 `run_v22.py`의 역사적 의미는 변경하지 않았다.
- 프로젝트 nnU-Net ResEnc M plan의 6단계 encoder 전체 + 공간 평균/128D projection. CT-only48³ 계약 유지, 불필요한 고정 그래프/45,385노드 보간 없음. 총90,337,920params; 입력·readout은 본 L0의 적용 설계로 full nnU-Net 재현 아님.
- 실제 liver_108의187개 관찰 위치를 모두 처리. 미학습 초기 가중치, optimizer0, L0-only. batch128 전체출력0.9435초/peak7.15GiB, 실제입력/배치/자동미분 확인. 큰배치187 실패와CUDA cache 수정도 보존. **전체 학습·평가 완료 아님.**
- [상세와 시각화 근거](docs/v22_cnn_l0_20260924.md), 최종 성공 결과 `work/v22_cnn_l0_one_case_20260924_release/`. 성장형 탐색은 연구 후보로 남아 있으며 이번 CNN을 성장형 그래프라고 부르지 않는다.

## 2026-09-23 이전 — 저장 공간 추가 압축 (연구 모델 버전 변경 없음)

- 현재 Basic CP NPY 캐시 1,076개에 NTFS 무손실 압축 적용. 전체 SHA/dtype/shape/mmap 검증, manifest 315개 보존. 원본·결과 삭제와 정밀도/모델 변경 없이 캐시 디렉터리에만 압축 상속 설정.
- 첫 정리 후 452.014 GiB였던 프로젝트 실제 파일 점유량이 **200.121 GiB**로 감소. 파일럿 포함 약 251.893 GiB 추가 확보. [측정과 검증 기록](docs/storage_compression_20260923.md).
- Basic CP 105 case manifest와 대표 3 case 실제 로딩, L0 14,102 patch 보존 확인. 전체 학습/평가 및 압축 후 학습 처리량 측정은 미실행.

## 2026-09-23 이전 — 데이터 내용 변경 없는 저장 공간 정리

- 사용자 정리 지시에 따라 동일 SHA1,713파일을 hardlink로 공유하고 폐기된 DEBUG/context NPY/NPZ11,870개를 삭제했다. 원본·현재 데이터·가중치·로그·지표는 보존. 실제 D 여유119.423GB 증가, 프로젝트 중복 제외 파일 크기452.014GiB로 감소.
- Basic CP105case/2,205배열 메타데이터와 대표3case 실제 로딩·SHA 검증, 현재 L0 patch14,102개 보존 확인. 모델/reader/정밀도는 변경하지 않았다. [정확한 삭제 범위와 검증](docs/storage_cleanup_20260923.md).

## 2026-09-23 이전 — v2.22 L0 U-Net형 경로 구현

- 사용자 요구를 단일 풀링으로 축소했던 구현을 정정했다. 새 graph_unet 경로는 두 단계 pooling과 mirrored unpool/skip, encoder3층/decoder2층으로 구성한다. 원래 노드 공간으로 복원한 후 L1/L2에 연결한다. 이전 late_sag는 단일 풀링 비교군으로만 남긴다.
- SAG/GAT U-Net 변형의 DEBUG scale45,385→6,912→1,728→6,912→45,385. 총1,949,659params. 원 Graph U-Nets의 A² 확장을 구현했다고 주장하지 않으며 중간 graph 단절을 감사에 기록했다.
- 실제 CT로 전체 loss/gradient/optimizer, 양쪽 skip/unpool branch,38테스트 검증. batch8 L0전체449.44ms/peak2.61GiB. 새 [상세 문서](docs/l0_graph_unet_20260923.md)와 `work/l0_graph_unet_debug_20260923/latency.csv`에 기록했다. 전체 학습/평가 미실행, r4 gate 유지.

## 2026-09-23 이전 — v2.22 L0 비교 구현: 조기 선택과 지연 풀링

- `early_ppr`: 첫 GAT 전에 선택. `early_sag`: 학습 선택 후 GAT3층. `late_sag`: 전체 GAT1층 후 학습 선택, 나머지2층. CNN-only/full graph 대조군과 공통 L1/L2를 연결했다. scorer gate가 query loss에서 학습되는 것을 검증했다. A*/방사형 강제 연결을 새 경로에 사용하지 않는다.
- 실제 CT와 전체 모델 규모로 DEBUG 비교. 별도 설정에서 유지1728/6912, batch2/4/8을 측정했다. K1728/batch8 L0 467.97ms(full),97.14ms(PPR),64.99ms(early SAG),212.45ms(late SAG). 학습 완료 성능이 아니다. 전체45,385후보 평가 비용을 포함한다.
- 34단위/회귀 검사와 실제 CT5구조 gradient/optimizer smoke 통과. 선택 graph NPZ, 연결성 감사, latency CSV/JSON을 저장했다. 초기 PPR 중심 편중과 SAG 연결 성분 분리도 기록했다.
- [구현 상세와 한계](docs/l0_comparison_implementation_20260923.md). 전체 코호트 학습/평가/production 승격은 미실행. 기존 v1/v2.1/v2.22 기록과 r4 차단은 보존한다.

## 2026-09-23 이전 — 실제 CT의 공간/특징 연결 시각화 진단

- 실제 liver_108 CT 패치와 미학습 CNN으로 전체45,385후보의 공간/특징 이웃을 표시하는 별도 진단 도구 추가. 8개 특징 이웃은 시각화 조건이며 최종 모델 설정이 아니다. CT 클릭,2-hop 확대/전체 보기,회전,두 종류 연결 표시를 검증했다.
- 초기 특징 연결 중6mm 초과는0.8221%로 대부분 공간 연결과 겹쳤다. 신규 L0 구현·학습 성공으로 간주하지 않는다. production 구조와 학습 차단은 그대로다. [선정 및 진단 기록](docs/v222_l0_method_selection_20260923.md).

## 2026-09-23 — r4 설계 반려와 학습 차단

- 실제 그래프의26개 고정 외곽 목적지가 방사형 연결을 강제했다. 경로/halo가 최종 노드의95.16~95.33%, 경로 길이가 직선거리의평균1.0056배다. 사용자 지적대로 PPR 선택보다 강제 경로가 지배한다.
- r4 학습 및 production checkpoint 사용을 명시적으로 차단했다. 코드·DEBUG·기존 결과는 보존하며 이전 구조로 복구하지 않는다. 설계 결함은 미해결이고 아래 실행 검증을 완성 근거로 취급하지 않는다.

## 2026-09-23 이전 — v2.22 r4 CNN-PPR/A* L0

- 사용자 제안한 개인화 PageRank 선택 + A* 외곽 경로를 실제 CNN→GNN→L1→L2 forward에 연결했다. 후보 전수 비용 평가, PPR 질량95% 선택,26개 외곽 목표 경로와1-hop 주변, 선택점의 기존 반경 엣지 전수 보존. 비용/95%/26방향/halo는 명시적 연구 선택이며 최적성 주장이 아니다.
- CNN/GNN/L1/L2 깊이·폭,1,519,063params,전체 데이터·split·seed42·CP80·epoch를 유지한다. 이전 r3는 ZIP/SHA로 보존. r3 graph/embedding/checkpoint는 r4로 재사용하지 않는다.
- 실제 CT3case/6위치에서 전체 후보45,385개를 평가. 최종8,934~8,969노드와345,574~347,382방향 엣지, 연결성, A*의 독립 Dijkstra 일치, 전체 모듈 gradient/optimizer 갱신 확인.60회귀 통과. [상세 계약·검증·한계](docs/v222_ppr_astar_r4_20260923.md).
- GPU backward에서 발견한 cluster-teacher autocast cache 문제 수정.8-corner 보간 streaming과 A* 동기화 제거.64batch 실측8.31graphs/s,peak6.32GB; 재확인7.74graphs/s.128batch 장시간 residency 지연은 실패로 보존하고 새 사전 검사에서 차단한다. 전체 모델 축소나 고정 batch cap 없음.
- 실제 그래프의 PPR/A*/halo/최종 연결을 조작해 볼 수 있는 로컬 화면과24보기 검증 추가. REFERENCES R18/R19 갱신. 전체 학습과 전체 평가는 아직 수행하지 않았다.

## 2026-09-23 이전 — 그래프 설계 감사·설명 정정 (모델 변경 없음)

- 현재 L0에는 그래프 샘플링 단계가 없음을 확인했다. 고정 격자 전체와 CNN 특징 보간을 샘플링 구조로 설명하지 않는다.
- 이전 context subset 함수의 존재를 요구사항 구현 완료로 취급한 설명도 정정한다. CT 가림·미완성 L1·표본 구성 비대칭과 subset 정책의 타당성 미검증 때문에 단순 복구하지 않는다.
- 보존본7파일 SHA, 과거 실제 CT DEBUG의 주요5소스 SHA 확인. 현재 모델 identity와 frozen v1 유지. [상세 감사](docs/v222_graph_design_audit_20260923.md).
- r3 학습 중단, 마지막 기록 step411/30000. CNN cell + feature-kNN 제안 철회. 새 모델·학습·전체 평가 없음.

## 2026-09-23 이전 — L0 설계 감사 (철회된 제안 포함)

- CNN12³ map을45385개 고정 노드로 보간하는 구조를 감사했다. Native CNN cell + layer별 특징 kNN 수정 명세와 DGCNN/ViG reference 추가. Node count/영역 계약이 바뀌므로 현재 학습에 적용하지 않음. 독립 진단4개 DEBUG 통과이며 새 모델 검증 완료가 아니다.

## 2026-09-23 이전 — v2.22 r3 VRAM 보정·실제 입력 시각화

- 후속 표시 수정: 실제 production `grid/edge`를 무손실로 내보내 전체 연결과1/2/3-hop을 표시한다. 모든 노드·엣지 복원 동일성, 브라우저 전체 표시 개수·회전·노드 선택·모바일 검증 완료. 고정 격자+CNN 특징 구조를 명시하며 학습 구조는 변경하지 않음.

- 16GB GPU에서 보정 중21.02/32.61GB를 실행하던 결함 수정: 실행 전 예측, 가용 VRAM 여유, 부적격 후보 중단, allocator 정리, 선택 batch 처리량 재검사. 첫 수정 뒤 전체 support11,279개16.784graphs/s 확인.
- 고정 support 메모리를 query batch마다 곱하던 과대 추정도 수정: 마지막 두 실측의 증가분으로 예측하고 증가분20% 여유 적용. 새6개/기존 관련17개, 총23회귀 통과. 모델·loss·노드·데이터·epoch는 유지.
- 실제 CT48³ 단면, GT 별도 overlay, 전체45,385개 노드, 선택 반경 이웃, L1/L2/CP 구조와 측정 로그를 확인하는 화면 추가. 학습된 attention/군집/성능은 아직 표시하지 않음.
- 이전 코드/실행/CT 캐시 보존. 새 전체 실행 `gnn_vram_affine`, 입력 provenance `index_vram_affine.json`. 전체 학습·전체 평가 미완료. [증거와 제한](docs/v222_visual_runtime_verification_20260923.md).

## 2026-09-23 이전 실행 — v2.22 r3 실행 오류 재점검·수정

- 사용자 자원 저활용/전체 구현 점검 요청으로 추가 결함을 확인했다. 서로 다른 case의 처리량을 비교한 CPU 보정을 같은 작업 묶음 비교로 수정했다.4worker 실측0.348case/s로1worker0.193 대비 약1.8배이며 새 전체 작업에서 실제4코어 사용을 확인했다.
- GAT의 과소 타일을 실측에 맞게 수정했다. 노드·엣지·모델 축소 없이 실제 기본 경로 batch16에서3.128graphs/s·peak7.472GB, 이전 batch4 약0.885 대비 약3.5배다.32batch는14.788GB이며 더 느렸다. 최종 full-support batch는 별도 보정한다.
- Frozen scorer가 nnU-Net RNG/backend 상태를 바꾸던 문제를 scope/복원으로 수정했다. Seed42·비교표본128개 계약 검사도 보강했다. 전체89검사 통과(1건 git 접근 문제는 임시 저장소 범위 설정으로 재검사).
- 원본131case 감사 완료,105학습case 모두128개 비교표본 확보, outer-val26은 표집 제외. 학습 입력11,279개와 inner-val2,823개를 유지한다. 새 실행 `work/v222_raw_ct_r3_training_20260923/`, Python PID6256(wrapper23268). 전체 준비 후40epoch GNN 실행이며 optimizer 시작/완료는 로그로 구분한다.250epoch segmentation과 전체 평가는 아직 미실행이다.
- [결함·수정·실측·현재 실행 보고서](docs/v222_runtime_review_20260923.md). 아래44검사·모델파일 바이트 동일 기록은 최초 r3 적용 당시 이력이며 이번 실행 보완은 별도 기록이다.

## 2026-09-23 — v2.22 r3 원본 CT·전체 공간 그래프

- 사용자 전체 수정안 승인 후 `hiercp_v222_raw_ct_cluster_r3` 적용. CT 중심 가림 제거, 기존 외곽 전체 보존 + 내부 일반 공간 노드, 비교 표본의 종양 거리 제한 제거. CNN/L1/L2·loss·CP 실행 규모는 유지한다. 이전 코드/문서는 `versions/v2.22/before_raw_ct_r3_20260923/` 보존.
- 회귀44검사·정적18파일 통과. 실제 CT liver_1/5/108, 전체45,385노드/2,383,482엣지에서 모든 파라미터 finite gradient·CNN/L0/L1/L2 optimizer 갱신과 중심/주변 특징 전달 확인. Batch2/4 peak1.065/1.982GB, 0.825/0.885 graphs/s. DEBUG이며 full training 성능이 아니다.
- liver_108 비교 후보는0개에서3,402,279개로 정상화,128개 선택 완료. 전체131case 새 사전 검사는 진행 중이며 완료 receipt는 `work/v222_raw_ct_r3_20260923/preflight/summary.json`이다.
- Basic CP80 입력105case 준비 완료 상태는 그대로다. 새 전체 GNN40epoch/nnU-Net250epoch/전체 평가는 미실행. 관측 점수의 CP 배치 효용은 미검증이며 case 독립성 한계와 whole-method 비교 범위를 유지한다.
- [현재 파이프라인](docs/pipeline_v222.md), [r3 검증 기록](docs/v222_raw_ct_r3_verification_20260923.md). 아래의 승인 대기·고정 가림 기록은 과거 이력이다.

## 2026-09-23 — 학습 목표 감사 및 원본 CT 수정 명세

- 가림 범위와 학습/CP 점수 경로의 읽기 전용 감사 도구를 추가했다. 원본 CT와 내부 일반 공간 노드를 복원하는 수정 명세를 작성했지만 학습 코드에는 적용하지 않았다. 자동 승인 차단 이후 전체 변경안 응답 대기 중이다.
- Basic CP80 준비105case 완료를 확인했다. 준비 완료와 GPU 학습 시작을 구분한다. [수정 명세·검증 계획](docs/v222_raw_ct_proposal_20260923.md).

## 2026-09-23 — v2.22 준비 병렬 측정 및 사전 검사

- 전체 case 표본 검사를 패치 생성 전으로 이동했다. 원래 거리 조건과 선택 난수는 그대로여서 liver_108 문제 자체는 아직 미해결이다.
- GNN 준비 전용 스케줄러를 연결해 동시 실행 수별 실제 처리량/RAM을 측정하고, 이후 작업을 연속 보충한다. 실행 전 admission과 측정별 receipt를 남긴다. 기존 frozen 버전과 실행 중 Basic CP 코드는 변경하지 않았다.
- 병렬/사전 검사8개 및 기존 모델/누수/군집24개 통과. 실제 CT CUDA DEBUG의 CNN/L0/L1/L2 gradient·optimizer 갱신과 batch2/4 측정 완료. 전체 학습·평가 미실행.
- 표본 정의 수정은 자동 승인 심사가 거부하여 정확한 변경안의 사용자 응답 대기 중이다. [상세 기록](docs/v222_preflight_parallel_20260923.md).

## 2026-09-23 — CP80 준비 재개 도구

- 완료 로그에 기록된30case를 실제 manifest/배열/원본 해시 검증 후 재사용하고 나머지75case를 기존 preparation 본문으로 처리하는 helper를 추가했다. 기존 실험 소스 identity·native manifest·데이터 계약은 변경하지 않았다. 부분 파일은 보존하고 다른 내용의 파일은 덮어쓰지 않는다.
- DEBUG 검사6개 통과. 사용자 재개 요청에 따라 `resume_20260923_1` 실행을 시작했다. 입력 준비 단계이며 모델 학습·평가는 아직 미실행이다. GNN 비교 중심 정의 변경은 미승인으로 보류한다.
- [재개 기록·체크리스트](docs/cp80_resume_20260923.md).

## 2026-09-22 22:40 KST — 사용자 요청 중단

- Basic CP80 준비 PID13468을 확인 후 중단했다. Wrapper22256도 종료됐다.105case 중30case 준비 기록과 모든 부분 파일·로그를 보존했다. GNN/nnU-Net 학습은 미시작이다.
- 사용자 재개 요청 전 자동 실행하지 않는다. 중단 기록: `work/cp80_comparison_20260922/stopped_20260922T134000Z.json`.

## 2026-09-22 CP80 비교 패치 — Basic CP와 v2.22 모두 시도 확률80%

- 사용자 지시에 따라 Basic CP80 전용 트레이너와 v2.22 config/bank/native 검증을0.8로 맞췄다. 원본 Basic의1.0 트레이너와 literal reference는 보존했다. Basic CLI 기본은0.8 비교이며1.0은 명시 인자로 선택한다.
- Seed42/동일 split/ResEncM/patch128³/batch2/250epoch를 비교 계약으로 기록했다. Gate도 공통 event 난수에 연결해 같은 학습 샘플에서 CP를 시도한다. 시도 실패를 재추첨하지 않으며 성공률을 시도율과 구분한다.
- 신규 gate6개 포함66검사 및 정적/frozen source 확인 통과. Basic 전체 입력 준비를 새 `work/cp80_comparison_20260922/`에 시작했다. GNN은 liver_108의 정답0 위치 정의 변경 답변 대기다. 전체 학습·평가·성능 비교는 미실행.
- [비교 계약·검증·체크리스트](docs/cp80_comparison_20260922.md); [정답0 위치 변경 제안](docs/cp80_comparison_center_decision.md). 변경 전 snapshot `versions/v2.22/before_cp80_comparison_20260922/`.

## 2026-09-22 실행 결과 정정 및 CP 확률 문헌 확인

- `case_benchmark_run1`은 준비 단계에서 `liver_108`의 적격 비교 중심0개 오류로 실패했다. GNN optimizer와 nnU-Net은 미실행. 부분 cache/오류 기록을 보존했으며 case 제외나 blind 규칙 축소를 하지 않았다.
- TumorCP 원문§3.2는 CP 확률0.8, object 변환 확률0.5를 구분한다. 원본 Basic CP의 매 방문 시도 또는 프로젝트 v2.22의0.5가 Copy-Paste 전체의 표준이라는 주장은 하지 않는다. References R13/R14에 추가했다. 이번 확인으로 학습 설정은 바꾸지 않았다.

## 2026-09-22 v2.22 r2 실행 패치 — 공개 case 기준 전체 GNN 학습 착수

- 사용자가 현 비교 조건으로 학습을 승인했다. CP0.5를 유지하되 최적 확률로 검증된 값이 아니라 기존 설정임을 명시했다. Basic CP의 매 방문 시도 정책은 보존했다.
- 공개 case benchmark provenance 형식을 추가했다. Case ID를 확인된 환자 ID로, 제공된 마스크를 완전한 주석으로 바꾸어 기록하지 않는다. 기존 verified-patient 계약은 그대로 엄격하게 검사한다.
- 전체 support memory의 첫 인코딩 전에 full-graph inference batch 및 worker를 실측한다. 최신 memory의 epoch 경계 중복 인코딩을 제거했으며 데이터·모델·그래프·40epoch는 줄이지 않았다.
- 새 경로 `work/v222_train_20260922/case_benchmark_run1/`에 전체 cache 준비→GNN40epoch 프로세스를 시작했다. 이 항목 작성 시점은 준비 단계이며 실제 optimizer 시작/완료 및 nnU-Net 학습과 구분한다. Native 단계는 GNN 완료 및 별도 처리량 보정 후 실행해야 한다.
- Case scope4개 포함 회귀43개·정적 검사 통과. [실행 범위·근거·체크리스트](docs/v222_case_training_run_20260922.md). 이전 소스는 `versions/v2.22/before_case_benchmark_training_20260922/`에 보존했다.

## 2026-09-22 v2.22 r2 추가 패치 — Basic CP 비교 시드 통일

- Basic CP의 unseeded/비결정적 native worker와 v2.22 seed42의 불일치를 수정했다. 두 경로가 `comparison_randomness.py`의 모델 초기화, ordered worker, epoch/worker/batch/phase RNG 일정을 공유한다. CP draw를 CT 선택·기본 증강과 분리했다. CLI 양쪽 `--seed` 기본42, `--workers` 필수이며 실행 receipt에 기록한다.
- 원본 Basic CP와 v2.22의 donor/크기/시도 확률/위치/crop 정책은 변경하지 않았다. 두 방법은 전체 파이프라인 비교이며 위치 선택만의 ablation이 아니다. 기존 데이터 분할/patch128³/batch2/250epochs 및 L0/L1/L2 구조는 유지한다.
- DEBUG 난수/native loader/실제 전체 ResEncM 초기 state_dict 검사9개, Basic CP8개, 모델 회귀39개 통과. Native CPU blur의 연산 경로별 반올림 차이는 허용 오차로 구분하며 GPU bitwise 재현성을 주장하지 않는다. 전체 학습·평가 미실행.
- [상세 감사와 완료 체크리스트](docs/comparison_seed_audit_20260922.md). 수정 전 소스는 `versions/v2.22/before_comparison_seed_fix_20260922/`에 보존했다. Source hash 변경으로 기존 preparation의 단순 재표기는 금지하며 새 준비가 필요하다.

## 2026-09-22 v2.22 r2 — 환자 간 prototype 군집 정렬과 References

- 현재 format: `hiercp_v222_cluster_alignment_r2`. r1 소스·문서는 `versions/v2.22/before_cluster_alignment_r2_20260922/`에 보존했다. L0·L1 구조/너비/깊이 및 전체 그래프 규칙은 유지했다.
- L2: query 환자 그룹을 먼저 제외한 뒤 실제 관측 근거가 있는 환자별 L1 label을 클래스별 cosine average-linkage로 군집화한다. 비단독 cut 전체의 양의 silhouette로 K를 선택하고 근거가 부족한 경우 K=1 사유를 기록한다. 환자/그래프를 버리는 cap은 없다.
- 고정 teacher center·assignment는 환자 episode마다 fit하고 batch에서 재사용한다. 기존 L2 2층/128/4 heads에 class-balanced alignment CE를 연결했다. Total loss는 query CE + alignment CE, CP 점수는 live L2 군집 중심의 환자 수 가중 log-sum-exp다. 모든 128개 후보를 점수화하는 online CP 흐름은 유지한다.
- 단독 환자의 미관측 class label은 prototype fit에서 제외한다. Query/validation 정답은 군집 fit에 넣지 않는다. Epoch 로그에 K 후보/선택 사유/점유 수/support 환자 목록/제외 query group/군집 안정성 ARI/중심 유사도·분산을 기록한다.
- 참고문헌을 `REFERENCES.md`에 별도 정리했다. SwAV, DeepCluster, PRODIGY, GATv2, silhouette, 계층 군집, nnU-Net 및 과거 검토 자료에 대해 실제 적용·프로젝트 변형·미사용을 구분한다. **SwAV 그대로의 재현이나 view-only L2라고 주장하지 않는다.**
- 군집 합성 DEBUG 검사 8개 추가, 기존 검사를 포함한 회귀 39개가 모두 통과했다. 정적 검사 18파일과 r1 대비 L0/L1 AST 보존도 확인했다. 실제 CT 3명/6개, 그래프당 24,174 nodes/1,229,900 edges, 파라미터 1,519,063개를 유지한 CUDA batch 2/4에서 gradient·optimizer 갱신을 확인했다. Peak CUDA 0.612/1.100GB. 실제 CT fixture는 support 2명/클래스라 K=1/1이고 다중 군집의 의료 효능 검증은 아니다.
- 전체 GNN 40-epoch·nnU-Net 250-epoch 학습 및 전체 평가 미실행. 실제 임상 군집/소형 종양 성능 개선은 미검증이다. Production identity/annotation manifest, 전체 가림 범위 감사가 여전히 필요하다. 기존 결과를 덮어쓰거나 학습 checkpoint를 만들어 내지 않았다.
- 상세: `docs/pipeline_v222_cluster_r2.md`; References: `REFERENCES.md`; 검증: `versions/v2.22/verification_cluster_r2_20260922/`; 최종 소스 해시와 일치하는 실제 CT 검증: `work/v222_20260922/cluster_r2_final_debug/result.json`. `code.txt`는 현 소스로 갱신하며 export receipt를 같은 검증 폴더에 기록한다. 아래 r1 이하 절은 보존 이력이다.

## 2026-09-22 v2.22 r1 — 관측 관계 학습과 누수 차단 구현

- 사용자 승인에 따라 독립 `hiercp_v222/`, `config/prompt_graph_v222.json`, `run_v222.py`를 구현했다. v2.21은 `versions/v2.21/before_v222_20260922/`에 보존했고 기존 버전 소스/설정은 변경하지 않았다.
- L0: CT 한 채널 CNN(12/24/32) + 공간 GATv2 3층/128차원/4 heads. 수작업 특징·종양 내부/정답 표면 노드를 입력하지 않는다. 학습 분할 전체로 정한 동일한 중심 가림을 보간 전에 적용하고 고정 물리 격자 전체를 사용한다.
- L1: 실제 주석으로 정의한 두 관측 클래스와 T/F support edge를 사용하는 2층 관계 attention. Query의 정답은 loss에만 전달한다. 환자별 자유 latent 16개와 관측 클래스 2개를 혼동하지 않는다.
- L2: query 환자 그룹 전체를 제외한 training support의 환자별 표현을 2층 cross-patient attention으로 정렬한다. Query CE가 L2/L1/query L0에 연결된다. Support L0 memory는 전체 inner-train에서 매 epoch 갱신하는 detached embedding이며 이 학습 선택을 문서화했다.
- 준비/40-epoch GNN 학습/frozen scorer/온라인 CP bank/250-epoch nnU-Net/predict/기존 v5 CSV 평가 CLI 경로를 연결했다. CP 후보 128개 전체 점수와 확률 0.5를 유지한다. 구/DEBUG checkpoint는 production bank에서 거부한다.
- 새 검사 12개 및 회귀 검사 총 31개 통과. 실제 CT 3개/6개 그래프의 결정론 CUDA DEBUG에서 전체 CNN/L0/L1/L2 gradient·optimizer 갱신 및 누수 개입 검사를 통과했다. 별도 native runtime의 V222 트레이너 import도 통과했다.
- DEBUG 그래프당 24,174 nodes/1,229,900 edges; 전체 파라미터 1,519,063개. 전체 그래프를 유지한 attention 계산 분할로 batch 2/4 최대 CUDA 할당량 0.605/1.091GB를 측정했다. Production batch는 full-cohort 자원 보정에서 정한다.
- **전체 데이터 준비·40-epoch GNN·250-epoch nnU-Net·전체 online CP/평가는 실행하지 않았다. 학습 checkpoint나 성능 결과도 만들지 않았다.** 본 준비에는 검증된 환자 그룹/주석 범위 manifest와 전체 가림 범위 감사가 필요하다. 점수는 관측 문맥 순위이며 보정된 종양 발생 확률이 아니다.
- 상세: `docs/pipeline_v222.md`; 검증: `versions/v2.22/verification_20260922/checks.json`; 실제 CT DEBUG: `work/v222_20260922/real_debug4_chunked/`.

# HierCP 패치 노트

## 2026-09-22 v2.21 작업 중 — T/F/U와 기하 조건 분리, PRODIGY 근거 정정

**v2.21은 아직 전체 모델 완성본이 아니다.** 기존 v2.2-r5는 보존했고 별도 `hiercp_v221/`, `config/prompt_graph_v221.json`, `run_v221.py`에 작업 중이다.

- 사용자는 T/F/U가 data-label 관계이며 F가 기하 불가능을 뜻하지 않는다고 정정했다.
- 관계 부여 기준 질문에 “원래 PRODIGY에서는 어떻게 했는데?”라고 답하여 원문과 공식 코드를 확인했다. 원문은 task class가 먼저 있고 support 정답 클래스=T/다른 클래스=F다. 난수는 label 초기 표현이며, 정답 없는 자유 latent slot 자체가 아니다. Query는 정답을 입력하지 않는다.
- 자기지도 Neighbor Matching도 그래프 이웃으로 임시 클래스/정답을 정의한 후 task를 구성한다. 이 관계가 CT 종양 발생 가능성의 정답이라는 뜻은 아니다. 논문 embedding은 256이며 현재 16/128의 근거로 삼지 않는다.
- 현재 구현 범위: 명시된 episode class에서 support T/F와 query U 및 방향/edge 속성을 생성하는 `relations.py`. 합성 DEBUG3개 통과. 기하 정보를 F로 변환하지 않는다.
- 의료 task/class 정의와 L1/L2 통합은 미완료다. 새 loss/teacher/클래스 부여 규칙을 임의로 추가하지 않았다. copied base를 v2.21 완성 모델로 실행하지 못하도록 차단했다.
- 상세 및 출처: [PRODIGY 관계 기준과 진행 상태](docs/prodigy_relation_basis_v221.md). `run_v221.py check`는 통과했지만 학습/평가 성공을 뜻하지 않는다.

## 이하: 이전 구현 상태의 보존 기록


## 2026-09-22 최신 정정 — 폐기한 종양 내부 노드 제거, data/label 근거 구분 (v2.2 r5)

**이 절이 아래 r4/r3의 “기존 종양 내부 유지”, “L0/L1 설계 확정” 설명보다 우선한다.** 사용자는 종양 내부 노드가 폐기된 설정이라고 명시했다. 과거 코드 보존을 이유로 활성 모델에 다시 넣지 않는다.

- `tumor_interior` 생성·세 관계 엣지·전용 projection/GNN/pooling을 제거했다. v2.2 전용 schema는 5종 node/13종 edge다. legacy 공통 checkpoint helper에 의한 재유입도 차단했다. 구 node가 들어간 cache/model 입력은 오류로 거절한다.
- surface/context CT-only CNN과 남은 GNN/L1/L2 깊이·hidden 폭은 유지했다. 별도 내부 그래프 노드 제거이며 CNN receptive field에서 내부 CT의 영향까지 없앴다는 뜻은 아니다.
- 사용자 원문에서 data는 L0가 인코딩한 실제 local context다. 현재 구현은 donor–candidate pair embedding을 data로 삼는다. 이 구체적 단위를 사용자 정의로 설명하지 않는다.
- label은 task별 독립 학습 가능한 잠정 latent 기준이다. 현재 L1 attention에는 원문의 T/F/U edge 정보가 없다. data별 T를 L2에서 집계한다고 data–label 관계 학습 계약이 구현 완료인 것은 아니다.
- label 개수 16은 기존 capacity를, 차원128은 기존 hidden_dim을 계승한 값이다. 사용자 지정값/생물학적 종류/최적값이라는 근거가 없다. 현재 projection/attention의 차원 호환과 연구적 타당성을 혼동하지 않는다. 임의 대체 수치를 추가하지 않았다.
- `l1_contract_status=unverified_pair_data_unit_and_missing_TFU_edge_conditioning`, label 개수/차원 rationale를 미검증으로 정정했다. 전체 학습은 미실행이며 학습 목적함수 문제도 남아 있다.
- 회귀 DEBUG16/16, 실제 inner-train CT2명/4graph CUDA BF16 batch2/4와 남은 L0/L1/L2 gradient/optimizer 경로 통과. 이것은 의미적 설계나 의료 성능 검증이 아니다.
- 보존본 `versions/v2.2/before_retired_interior_removal_r5_20260922/`; 증거 `versions/v2.2/verification_no_interior_20260922/`; 현재 format `hiercp_prompt_graph_v22_surface_context_cnn_r5`.
- [정의·근거·폐기 설정 상세](docs/design_contract_corrections_v22_r5.md). code.txt와 패치노트를 같은 소스로 최신화한다. 전체 학습, GitHub push, 서버 배포는 하지 않았다.

## 이하: 이전 버전·설명 보존 기록

## v2.2 패치 — 2026-09-22 정정 — CNN L0/L1 유지, 기존 L2 복구 (v2.2 내부 r4)

**이 절이 아래 r3의 L2 삭제·미구현 설명보다 우선한다.** L0 특징 단순화는 L2 삭제를 허가한 요구가 아니었다. 이전 코드에서 L2는 L1의 128차원 label만 받으며 수작업 특징을 직접 요구하지 않는다. 통계 기반 보조 loss와 L2 모듈을 혼동하여 함께 삭제했던 오류를 수정했다.

- `PromptGraphModel`에 기존 `CrossPatientAlignment` 2층/128/4 heads, 환자 간 label 대응과 외부 관측 T 전달을 복구했다. L0는 CT-only CNN 특징32, L1은 기존2block/16labels로 유지한다. 두 모듈 모두 가중치 학습 가능하다.
- L2 클래스·증거 전달·기존 관측 loss는 r2 보존본과 AST 동일하다. 전체 forward와 `forward_tasks`가 작동하며 CNN부터 L2까지 기존 관측 loss로 gradient와 optimizer update가 확인됐다.
- 통계 descriptor/복원 decoder/통계 유사도 teacher는 복구하지 않았다. **L2가 없거나 역할이 미정인 것이 아니다.** 전체 학습 목표가 통계 보조 loss 제거 이후 미완성인 것이다.
- 현재 준비 데이터는 T/U이며 F는 hard geometry gate에서 제외한다. 균등 label/대응 분포가 모든 후보 score=1, 관측 loss=0을 만족하는 반례를 검증했다. 따라서 기존 관측 loss만으로 전체 학습을 임의 재개하지 않는다. U를 F로 바꾸거나 새 teacher/loss를 추가하지 않았다.
- 단위/회귀 DEBUG 15/15, 실제 inner-train CT2명/4graph, CUDA BF16 batch2/4, L2 두 층을 포함한 주요 그룹 갱신 통과. 전체 학습·평가·소형 종양 성능 검증은 미실행이다.
- 증거: `versions/v2.2/verification_l2_restore_20260922/`. 수정 전 r3는 `versions/v2.2/before_l2_restore_r4_20260922/`에 보존했다. 현재 format은 `hiercp_prompt_graph_v22_ct_cnn_l0_l1_l2_r4`다.
- 상세: [L2 복구와 설계 결함 감사](docs/pipeline_v22_l2_restoration.md). code.txt도 현 소스로 재생성한다. GitHub push/서버 배포는 수행하지 않았다.

## 이하: 과거 구현·판단 보존 기록

## v2.2 — CT-only CNN L0 · L1 구조 확정 (2026-09-22, 내부 r3)

사용자 요청에 따라 통계 특징을 늘리던 r1/r2를 보존하고 CNN-only 노드 특징으로 변경했다. CT 1채널, CNN 출력 32차원, 추가 node 특징 0, edge 속성 0. 기존 graph 규칙·GNN 폭/깊이·L1 구조는 유지한다. 구조 고정이며 가중치는 학습 가능하다.

수작업 통계 복원과 L2 teacher를 활성 모델에서 제거했다. 이번 완료 범위는 L0/L1이며 L2 대체 objective는 미확정이다. 전체 학습과 CP 점수 생성은 명시적으로 차단한다. 기존 목표를 CNN 목표로 조용히 치환하지 않았다.

## 2026-09-22 완료된 검증

- 정적 검사 25개 파일, 새 단위/회귀 DEBUG **8/8 통과**. v1/v2.1 runtime/config SHA guard 통과. L1 `Residual`/`PatientTaskGraph` AST가 보존본과 동일하다.
- 실제 inner-train `liver_1`, `liver_5`에서 T/U 각 1개, 총 4개 graph를 DEBUG로 검사했다. canonical node/edge와 sampled node/edge가 기존 규칙과 정확히 일치하고 CT는 기존 첫 채널과 일치했다. 원래 128개 후보 생성 규칙은 유지했다.
- CUDA BF16에서 CNN·GNN 3층·L1 양방향 2 block을 포함한 gradient가 모두 유한했고 실제 AdamW DEBUG update가 확인됐다. 환자 label table 2개를 둔 DEBUG 모델의 파라미터 **6,453,844개**가 전부 학습 가능하다. 전체 파라미터 수는 등록 환자 수에 따라 달라진다.
- DEBUG physical batch 2: 1.871 graph/s, peak allocated 264.5 MiB. batch 4: 2.689 graph/s, 486.2 MiB. warm-up 후 각각 3회 forward/backward. production L2/cohort support가 없는 L0/L1 probe이므로 최종 학습 batch/속도/VRAM으로 일반화하지 않는다.
- 검증 근거: `versions/v2.2/verification_cnn_only_20260922/implementation_checks.json`, `real_CT_cuda_debug.json`, `geometry_debug.json`. 원 실행 로그 데이터: `work/v22_cnn_r3_20260922/debug1/`.
- 전체 GNN/nnU-Net 학습·전체 평가·의료 성능 검증은 미실행. L2 학습 목표는 미확정이다.

상세 [L0/L1 계약](docs/pipeline_v22_cnn_l0_l1.md). 변경 전 보존본은 `versions/v2.2/before_cnn_only_r3_20260922/`다. 기존 체크포인트/캐시는 새 버전으로 재명명해 쓰지 않는다.

## 이하: 과거 v2.2 r1/r2 기록 (현재 구현 아님)

## 2026-09-22 근거·상태 정정 (새 모델 버전 아님)

- [특징·L2 근거 대조](docs/feature_evidence_audit_20260922.md)를 추가했다. 사용자 첨부에는 통계 특징의 제안이 있지만 현재 16/24차원 조합의 검증 근거는 아니다. SPG의 다른 과제 기하 특징 효과, IBSI의 계산 정의, 현재 CP 직접 근거를 구분했다.
- L2의 descriptor 유사도 teacher는 사용자 원문의 정렬 목적이나 PRODIGY의 구체적 학습식에서 도출된 검증된 구현이 아니다. 임의 대체 모듈/특징을 추가하지 않았다.
- 아래 v2.2 설명과 DEBUG 기록은 r1 당시 기록으로 보존한다. 현재 로컬 소스는 미완성·중단된 r2(24차원 특징/144차원 descriptor)이며 r1 검증이 r2 완료를 의미하지 않는다. `gpt_handoff.md` 상단에 이 상태를 정정했다. 기존 code.txt 및 상세 r2 인계 문서 전체 동기화는 완료하지 않았다.
- 이번 변경은 문서뿐이다. 학습·평가·테스트를 재개하지 않았고 기존 모델/설정/실험 결과를 변경하지 않았다.

## 이하: 2026-09-20 기록 보존

현재 구현 버전: **v2.2** · 갱신: **2026-09-20 KST**. 전체 학습 완료를 뜻하지 않는다.

이 문서는 설계·구현 변경 이력의 기준이다. 버전 번호는 로컬 파이프라인을 구분하며 GitHub release/tag 생성, 서버 배포 또는 전체 학습 완료를 뜻하지 않는다.

| 버전 | 상태 | 무엇을 했는가 |
|---|---|---|
| v1 | 기존 기준선 보존 | 기존 L0 국소 그래프, L1 patient/region, L2 population/prototype 및 feedback CP 파이프라인 |
| v2.0 | 잘못된 view-only 설계, 폐기·원본 보존 | 동일 데이터의 두 stochastic view에 각각 label을 만들고 view 간 일치만 학습 |
| v2.1 | 기존 구현·실행 보존 | 실제 환자별 L1의 임의적 label 공간을 L2에서 정렬하고 다른 환자의 관측 positive evidence 전달 |
| **v2.2** | **CNN 없는 경로 구현·DEBUG 검증** | 기존 공간 그래프를 유지하고 CT·통계·기하 특징 → MLP → GNN으로 학습. 별도 인코더·checkpoint·bank |

상세 구조는 [v2.2 파이프라인](docs/pipeline_v22.md), [v2.1 파이프라인](docs/pipeline_v2.md), 인계 상태는 [gpt_handoff.md](gpt_handoff.md), 보존 위치는 [버전 안내](versions/README.md)를 따른다.

## v2.2 — CNN 없는 원본 관측 특징 경로, 2026-09-20

- `hiercp_v22/`, `run_v22.py`, `config/prompt_graph_v22.json`으로 분리했다. v2.1 source/config/CLI 및 이전 handoff/code.txt를 `versions/v2.1/before_v22_20260920/`에 SHA와 함께 보존했다. 기존 runtime와 실험 결과는 변경하지 않았다.
- 별도 3D CNN·feature-map sampling을 제거했다. 원본 16개 관측 특징을 node별 MLP와 기존 3층 gated GNN이 학습한다. hidden 128, head 4, L1/L2 2/2층, label 16, donor·후보·분할·epoch 조건 유지.
- GPU 배치와 새 graph 저장 파일에서 48³ dense patch를 제거했다. L2 descriptor 목표는 기존 패치의 mean/std 20개로 정확하게 보존한다. CPU에서 geometry/패치 통계를 만드는 비용은 남는다.
- 모든 geometry를 유지하는 `convert-cache`, 별도 결과 경로로 전처리를 재사용하는 `reuse-native` 추가. 부분 cache와 v2.1 가중치의 자동 재사용은 거부한다.
- 회귀 32개와 CUDA full-width gradient/optimizer DEBUG 통과. 실제 그래프 4개의 tensor·descriptor 동일성 확인. batch 2/4 비교에서 v2.1 CNN은 약 5–6 ms였으며 그래프 연산 비용이 더 컸다. CNN이 전체 병목이었다거나 의료 성능이 좋아졌다고 주장하지 않는다.
- 실제 128개 후보 전체의 CNN-free Scorer 경로와 argmax를 검증했다. 실제 T anchor 3명, 미학습 모델의 DEBUG이며 전체 학습 또는 새 native paste 실행 결과가 아니다. 검사 도구의 첫 JSON 기록 오류를 수정한 뒤 `pool_debug2/verification.json`에 성공 증거를 남겼다.
- T-only 점수 학습·L2 통계 정렬·학습/사용 context 차이는 기존 미검증 과제로 남는다. 전체 학습·의료 평가는 미실행. 자세한 근거와 실행 방법은 v2.2 문서를 따른다.

## v2.1 — 실제 환자별 L1 / 환자 간 L2 정렬

### 2026-09-19 사용자 정정: Basic CP는 원본 정책을 유지하고 온라인화만 수행

- 기존 실험용 Basic이라는 이름에 들어 있던 ≤20 mm source 제한과 128개 후보 풀은 원본 Basic CP의 규칙이 아니다. 공통 donor 풀도 원본 규칙이 아니다. 과거 결과를 원본 Basic 결과로 재표기하지 않는다.
- `basic_cp_online/`과 `run_basic_cp_online.py`를 새로 분리했다. 자기 환자의 전체 종양 중 균등 무작위 1개, 무작위 탐색 첫 유효 위치, 원본 4,000회 한도, 1회 paste, 원본 HU jitter를 유지한다. 추가 50% 적용 gate는 없다. 학습 case 방문마다 새로 추출하며 native nnU-Net crop과 validation 경로를 유지한다.
- 원본 uint8 mask inversion 거리 오류는 literal 원본 재현 경로에 명시적으로 보존한다. 오류 수정 실험과 원본 온라인화 실험을 섞지 않는다. 12개 seed의 원본 실행 CT/GT 대조, native 변환과 실제 표준 loader를 포함한 DEBUG 8개가 통과했다. 전체 학습은 미실행이다.
- 상세 계약과 실행·미검증 범위: [원본 Basic CP 온라인](docs/original_basic_cp_online.md). 기존 v1/v2 런타임과 실행 중인 GNN 코드는 변경하지 않았다.
- 실제 `liver_2`의 자기 종양 25.3166 mm / 14,131 voxel을 축소 없이 증강했고, 원본 스크립트 실행과 전체 raw CT/GT 완전 일치·원본 SHA 불변을 확인했다. 이 환자는 기존 ≤20 mm 변형에서 제외됐지만 원본 Basic에서는 제외되지 않는다. `work/original_basic_cp_online_20260919/debug1/`에 현재 소스 SHA의 DEBUG 증거를 보존했다. 전체 Basic 학습·native 의료 평가 완료를 뜻하지 않는다.

### 2026-09-19 추가 패치: 모든 학습 환자의 공통 donor 증강과 분할 누수 방지

- 소형 종양 조건을 donor 자격에만 적용한다. outer-train 105명 모두 동일한 inner-train 적격 donor 527개에 접근하며, native CP 이벤트마다 균등 선택한다. 원래 소형 종양이 없는 환자도 제외하지 않는다. CP 확률 0.5는 유지한다.
- 잘못된 81명 자기 donor / 24명 전체 donor 분기를 제거했다. GNN context는 실제 T anchor 662개 전부와, donor·recipient를 모두 포함하는 균형 round의 128개 후보로 구성한다. train 67,983개 + validation 67,591개 = 총 135,574개다. native CP의 이벤트별 전체 풀 추출과 별도이며 환자별 donor를 고정하지 않는다.
- native bank는 전체 recipient/donor catalog를 검증한 뒤, 실제 요청된 pair에서 전체 128개 graph와 점수를 계산한다. 고정 argmax의 실제 raw CT/GT paste 한 개와 전체 점수·좌표를 저장한다. 원본 CT/GT는 수정하지 않는다. preparation subtree를 runtime bank 재로딩 시 복원하도록 저장 경로를 보완했다.
- donor는 inner-train만, GNN gradient·support도 inner-train만 사용한다. outer-val은 CP recipient에도 넣지 않는다. nnU-Net validation은 기존 일반 loader를 유지한다. split 중복, 분할을 넘는 동일 원본 영상, donor membership, checkpoint support IDs 및 원본·plan·catalog·payload SHA를 검사한다. 단계별 경계와 검사 한계는 [누수 방지 계약](docs/shared_donor_leakage_v21.md)에 명시했다.
- 작업 queue를 연속 배정하고 실제 측정에 따른 CPU/RAM admission과 공유 volume cache를 연결했다. L1의 모든 context를 유지하면서 activation checkpointing과 attention 실행 분할을 적용해 CUDA kernel의 65,535-row 제한을 해결했다. batch/worker는 실제 후보 측정으로 선택하며, 과대 allocation 전 측정 메모리 증가율을 검사한다. 모델·graph·후보·40/250 epoch는 유지한다.
- 수정 전 17개 소스는 `versions/v2/pre_shared_donor_fix_20260919/`에 보존했다. 이전 비대칭 cache/bank/checkpoint는 재사용하지 않는다. 회귀 39/39 통과. 실제 CT의 128후보→native paste 검증 및 전체 학습 상태는 [로컬 실행 기록](docs/local_v21_5070ti_run.md)을 따른다. DEBUG 검사 통과를 의료 성능이나 전체 학습 완료로 보지 않는다.
- 실제 외부 donor를 원래 적격 source가 없는 `liver_2`에 적용하는 DEBUG 연결 검사도 통과했다. 128개 전체 점수 argmax에 CT/GT 3,165 voxel을 paste했으며 영역 밖 GT·원본 SHA를 보존했다. 미학습 모델을 사용한 검사로 의료 성능 결과는 아니다.
- 22:21 KST에 `attempt3_shared_donor/`에서 전체 준비→GNN 40 epoch launcher를 시작했다. 원본 131명만 재사용하며 이전 부분 graph cache는 재사용하지 않는다. 현재 소스 SHA의 최종 정적 22개·회귀 39개·CUDA optimizer·67,983 context 역전파·실데이터 native paste 증거를 `versions/v2/verification_shared_donor_20260919/`에 보존했다. 학습 완료나 nnU-Net 250 epoch 실행 완료를 뜻하지 않는다.

### 2026-09-19 실행 중단 정정: donor 유무에 따른 후보 정책 불일치

사용자 지적에 따라 실제 분기를 확인했다. 소형 source가 있는 81명은 자기 환자 source만 사용하지만, 없는 24명에게만 전체 inner-train source 527개×128개 후보를 적용한다. 예: liver_1은 11개 source/1,419 graphs, liver_3은 1개/129 graphs, liver_2는 자기 source 0개인데 외부 527개/67,456 graphs다. 모든 환자에게 같은 donor 정책을 적용한 것이 아니며, L2 정렬의 필수 조건으로 설명할 근거가 없다. 이 정책을 확정된 설계로 실행한 점을 정정한다.

해당 graph 준비 프로세스 PID 37104만 중단했다. 전체 GNN optimizer는 미시작이며 부분 cache는 학습에 사용할 수 없다. 원본 131명 데이터·기존 결과·완료된 native 131명 전처리는 보존했다. 아직 후보 정책이나 모델 규모를 변경하지 않았으며 자동 재실행하지 않는다. 실행 중단 기록은 `work/local_v21_5070ti_20260919/attempt2/stopped_donor_policy_review.json`이다. 아래 실행 중이라는 설명은 중단 이전 이력이다.

### 2026-09-19 추가 패치: 전체 의료 데이터 준비와 실데이터 CUDA 검증

- MSD Task03 Liver archive 28,925,891,584 bytes의 다운로드·공식 MD5 검증·추출을 완료했다. 정답이 있는 131명 전부의 CT/GT를 검증했으며 outer train/val 105/26, inner train/val 84/21이다. 서버 실험과 분할 byte 동일성을 확인한 것은 아니다.
- 첫 준비 실행은 105명 중 37명의 source inventory 처리 후 RAM admission `MemoryError`로 중단됐다. 실패 로그와 부분 출력은 `work/local_v21_5070ti_20260919/`에 남겼다. GPU 학습 OOM이나 학습 완료로 기록하지 않는다.
- 환자별 모든 종양 mask를 동시에 보관하던 코드를 반복 가능한 lazy source collection으로 바꿨다. 후보별 공통 source tensor는 SHA로 식별해 한 번 저장하고 나머지는 gzip으로 무손실 저장한다. 원래 tensor와 로딩 결과의 정확한 일치를 검사했다. 모델·환자·병변·후보 수는 변경하지 않았다. 수정 전 원본은 `versions/v2/pre_storage_fix_20260919/`에 보존했다.
- 전체 원본의 기존 6-connectivity/≤20 mm 규칙에서 inner-train donor source는 527개다. 적격 T가 없는 outer-train 환자 24명도 모든 donor의 128개 후보를 쓰므로 예정 cache는 **1,704,342개 graph**다. 전체 cache 용량·소요시간과 이 규모의 L1/L2 GPU 메모리는 아직 검증되지 않았다.
- `attempt2/`에서 완료된 131명 원본 데이터의 SHA를 다시 확인하고 전체 graph 준비→GNN 40 epoch 실행을 시작했다. nnU-Net train-only fingerprint/planning은 완료됐고 131명 전처리를 별도로 실행했다. GNN optimizer 시작과 native 학습 완료는 별도 단계다.
- 새 source identity에서 정적 검사 17개, CPU/CUDA 회귀 **26/26 통과**, 실패·skip 0개. 실제 CT 환자 3명의 full-scale graph 6개로 **실데이터 DEBUG BF16 optimizer 1 step**도 통과했다. loss 3.203799247741699, 모든 gradient 유한값, L0/L1/L2를 포함한 핵심 모듈 갱신, peak allocated VRAM 1,086,012,416 bytes. DEBUG subset이며 checkpoint를 생성하지 않았다. 전체 학습 결과나 의료 성능 점수가 아니다.
- 최신 증거는 `versions/v2/verification_storage_real_cuda_20260919/`, 상세 단계·제한은 [로컬 실행 기록](docs/local_v21_5070ti_run.md)을 따른다. 아래 문서 정리 당시의 CPU-only/미실행 설명은 역사적 상태다.

### 2026-09-19 추가 패치: RTX 5070 Ti 실제 CUDA 실행

- 프로젝트 전용 `.venv`에 PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128 / PyG 2.6.1 / nnU-Net 2.8.1을 설치했다. 재설치 시 `config/runtime_windows_v21.txt`의 제약을 적용한다. 시스템 드라이버와 기존 Python 환경은 변경하지 않았다.
- 첫 실제 CUDA BF16 검사에서 L1의 `index_add_` 버퍼(BF16)와 weighted message(FP32) 자료형 불일치가 발견됐다. 버퍼를 곱셈 결과의 승격된 dtype으로 만들도록 수정했다. 레이어·채널·그래프·학습 규모와 L1/L2 의미는 유지한다.
- 수정 전 파일은 `versions/v2/pre_cuda_dtype_fix_20260919/`에 보존했다. 런타임 소스 SHA가 바뀌었으므로 과거 source identity를 가진 checkpoint의 자동 재사용은 허용하지 않는다.
- 실제 5070 Ti에서 7,086,156-parameter DEBUG 모델의 BF16 forward/backward/optimizer 1 step 통과. 회귀 **26/26 통과**, 실패·skip 0개. 기존 25개 CPU 검사와 구별한다. 이 합성 검사는 의료 성능 결과가 아니다.
- 공식 MONAI 배포 MSD Task03 Liver 전체 archive 다운로드·체크섬 검증, 환자 131명 검증 및 기존 stratified split 재구성 도구를 추가했다. 서버 원본 split/hash 동일성은 미확인이다.
- 실행 절차와 진행 상태는 [로컬 실행 기록](docs/local_v21_5070ti_run.md)을 따른다. 전체 학습 완료와 nnU-Net 성능 평가 완료는 해당 완료 파일이 생긴 뒤에만 기록한다.

기록일: 2026-09-19. 기존 문서의 `cross-patient r1` 구현을 **v2.1**로 명명한다. 이번 패치 노트 정리는 버전 명칭·문서 동기화이며 모델을 다시 변경하거나 학습한 작업이 아니다.

### 수정한 문제와 구현

- **L0:** 기존 KD-tree 질의로 만드는 국소 physical graph와 GNN 인코더를 유지한다. 그래프 구성과 그래프 인코딩은 별개 단계다.
- **L1:** task를 실제 patient ID로 정의한다. 각 환자마다 독립적으로 초기화한 학습 가능한 label nodes를 두고 data↔label 관계를 학습한다. 이 label은 국소적·임의적 표현이며 절대 정답이나 모든 환자에게 공통인 class ID가 아니다. K-means label과 view를 task로 취급하던 방식을 제거했다.
- **L2:** 환자별 L1 label 공간의 대응 관계를 학습한다. 다른 inner-train 환자의 관측 T를 target 환자의 공간으로 전달해 U 후보와의 compatibility를 계산한다. target 자신의 T와 inner-val 환자를 evidence donor로 사용하지 않는다. T가 없는 환자도 U-only task로 참여한다.
- **T/F/U:** T는 실제 적격 tumor source 원위치, F는 명확한 기하학적 불가능, U는 유효하지만 positive가 관측되지 않은 위치다. U를 negative로 감독하지 않는다. 생산 cache는 T/U를 만들며 불가능한 후보는 geometry 검사에서 제외한다. 점수는 암 존재 확률이 아니다.
- **학습 연결:** 관측 T/F 오차, 실제 context 통계 reconstruction, 환자 간 soft alignment를 연결했다. 통계 유사성을 correspondence teacher로 사용하는 것은 구현의 연구 가정이며 의학적 정답으로 검증된 것이 아니다.
- **CP·평가 연결:** frozen GNN의 `cross_patient_positive_transport_argmax`로 후보를 선택한다. 실제 donor paste에 대해서만 synthetic segmentation GT를 만든다. 크기별 지표는 `size_metrics.csv`, 개별 병변 지표는 `lesion_metrics.csv` 등에 기록하도록 구현했다.

### 유지한 규모와 호환성

L0 hidden 128 / heads 4 / layers 3, L1/L2 layers 2/2, CNN 12/24/32, patch 5×48³, 환자별 label 16×128, GNN 40 epoch, nnU-Net ResEncM 250 epoch, source당 후보 128개, CP 확률 0.5를 유지한다. label 16개는 구현의 capacity 설정이며 확정된 의학 범주 수가 아니다.

현재 v2.1의 내부 artifact format은 **`hiercp_prompt_graph_v2_cross_patient_r1`**이다. `run_v2.py`, `hiercp_v2/`, `config/prompt_graph_v2.json`, `versions/v2/`의 경로명은 v2 계열 경로로 유지한다. 경로에 `v2`가 있다는 이유로 폐기된 v2.0으로 해석하지 않는다. 내부 format 문자열을 이름만 바꿔 checkpoint를 이전하지 않는다.

v1·v2.0 cache/checkpoint/bank는 v2.1과 호환되지 않는다. 새 출력 디렉터리와 새 데이터 준비·학습이 필요하다. 자동 resume와 DDP는 미구현이다. v1의 별도 feedback curriculum과도 다르므로 v1 대비 차이를 L2 단독 ablation 효과로 보고하면 안 된다.

### 검증과 미실행 항목

기존 [검증 기록](versions/v2/verification_cross_patient_final_20260919/verification.json)과 [테스트 로그](versions/v2/verification_cross_patient_final_20260919/tests.txt): **합성 DEBUG 25/25 통과, 실패·skip 0개, Python 16개 정적 검사 통과**. 환자별 label 독립성, U supervised gradient 배제, 외부 T 전달, 자기 T 제외, label 순열, full-width gradient/optimizer 연결 등을 검사했다. 준비-loop의 128개 후보 검사는 geometry를 mock한 정책 검사이며 실제 의료 데이터 검증이 아니다.

**실제 의료 데이터 전체 GNN 학습·nnU-Net 학습·native end-to-end 실행·전체 의료 평가·target GPU 처리량 측정은 미실행이다. v2.1의 실제 성능 점수나 학습 완료 가중치는 없다.** 이번 문서 변경에서는 모델 테스트를 재실행하지 않고 기존 검증의 소스 SHA 일치와 문서 export를 확인한다.

## v2.0 — 잘못된 view-only 구현, 폐기

이력 구분일: 2026-09-19. 이전 문서에서 처음 `v2`라고 불렀던 구현이다.

동일 데이터에 두 stochastic L0 view를 만들고 각 view에 K-means label을 부여했다. L1은 data–label 관계를 만들고 L2는 shared-support contingency / view consistency로 두 view를 맞췄다. 정렬 embedding의 cosine 기반 후보 선택을 사용했다.

**잘못된 점:** 사용자가 의도한 L2는 서로 다른 실제 환자의 국소 임의 label 공간을 정렬하는 역할이다. 같은 데이터의 view 일치만으로 대체한 v2.0은 이 요구를 충족하지 못했다. 현재 설계나 최종 baseline으로 사용하지 않는다.

- 소스·당시 handoff·code.txt **218개**: [보존 ZIP](versions/v2/superseded_view_alignment_20260919/source.zip), [SHA manifest](versions/v2/superseded_view_alignment_20260919/manifest.json).
- 과거 내부 format: `hiercp_prompt_graph_v2`. v2.1 format과 구별한다.
- `versions/v2/verification_20260919/`와 `verification_handoff_20260919/`의 테스트는 당시 구현의 역사적 검사다. 이를 v2.1 검증이나 의료 성능 결과로 인용하지 않는다.

## v1 — 기존 HierCP 파이프라인 보존

보존 기준 commit: `74dcc2cf03d2d40d1f582223321d96004333f661`. 버전 정리일: 2026-09-19. 이 날짜는 원래 실험 수행일이 아니다.

- 기존 `run.py`, `hiercp/`, `custom_trainers/`, 기존 설정과 실험 경로를 보존했다. `run_v1.py`는 기존 진입점의 별칭이다.
- L0는 기존 국소 그래프 인코더, L1은 patient/region 계층, L2는 population/prototype 계층이다. 기존 online CP와 feedback/curriculum 구현을 유지한다.
- 내부 `MODEL_ARCHITECTURE_VERSION=v5`는 **파이프라인 v1** 안의 모델 revision이다. 파이프라인 v5로 재명명하지 않는다.
- 추적 파일 **202개**를 [원본 ZIP](versions/v1/pipeline_v1_source.zip)과 [SHA manifest](versions/v1/manifest.json)에 보존했다. 현재 handoff/code.txt는 갱신하되 두 파일의 v1 원문은 ZIP에서 검증한다. 다른 원래 runtime/config 파일은 현재 파일도 원본 SHA와 일치해야 한다.
- 과거 성능·ablation 결과는 원래 실험명과 평가 기준을 유지한다. [과거 결과 정리](docs/results_summary_20260918.md)와 `experiment_results/`를 참조하며 v2.0/v2.1 성능으로 옮겨 적지 않는다. 이번 버전 정리는 기존 실험 재실행이 아니다.

## 이후 기록 규칙

변경할 때마다 이 파일 상단에 새 버전 항목을 추가하고 **날짜, 수정한 문제, 구조·코드·설정 변경, 호환성, 검사 근거, 미실행 항목**을 기록한다. 이전 항목과 보존 ZIP은 덮어쓰지 않는다. 수정된 설명이 필요하면 정정 내용을 별도로 남긴다.

현재 버전 표기는 이 문서·`versions/README.md`·`docs/pipeline_v2.md`·`gpt_handoff.md`에서 함께 갱신하고 `code.txt`를 재생성한다. 실행 파일명과 저장 format ID는 사람이 읽는 버전 번호와 구별한다. DEBUG 통과와 실제 전체 학습·평가, 로컬 변경과 GitHub 배포를 각각 기록한다.

## 작업 완료 체크리스트

이번 작업은 패치 노트 및 인계 문서 정리다. 아래 미체크 항목은 이번 문서 작업에서 수행하지 않은 실행 검증이다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 구현 상태만 기록했으며 새 측정은 하지 않았다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 검증 기록을 참조했으며 새 자원 측정은 하지 않았다.
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 작업에 OOM 실행은 없었다.
- [x] 디버그 설정과 최종 설정을 분리해서 기록했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 DEBUG 증거를 인용했으며 이번에 재실행하지 않았다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.


## 작업 완료 체크리스트


- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] 메모리 문제에 그래프 축소 없이 기존 tiling/checkpointing 경로를 유지했다. 새 DEBUG에서 OOM은 없었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
