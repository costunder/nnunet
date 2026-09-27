# v2.2 온라인 추천 연결과 캐시 저장량 검사

## 수정 범위

`ea702fd`의 core 83개와 학습 runtime 20개는 유지한다. 학습 모델·loss·규모·학습 checkpoint 계약은 변경하지 않았다. 새 온라인 모듈과 별도 trainer를 추가했으며, 캐시 준비 실행기에는 저장량 검사를 연결했다. 이전 `hiercp_v222.bank/native_adapter` 경로를 최신 ranking 경로라고 사용하지 않는다.

## 실제 연결

`tools/v22_online_rank_bank.py`에서 검증된 final ranking artifact를 읽는다. inner-train donor 풀과 outer-train recipient를 유지하고 기존 난수 draw로 donor를 선택한다. 동일 환자 donor가 뽑히면 다시 뽑지 않고 원본을 유지한다. 이는 추천 API의 self-patient 금지를 지키면서 전체 donor draw 분모를 유지한다.

후보 생성은 기존 간 union 및 물리 기하 규칙을 사용하되 종양 주석에 따른 선제 제외를 제거했다. 128개가 만들어지지 않으면 축소하지 않고 오류를 낸다. 관측 종양 anchor를 점수 1위로 강제하거나 점수를 변경하지 않는다. 종양을 간으로 바꿔도 간 union이 같으면 proposal이 같다는 DEBUG 검사를 통과했다.

실제 donor와 recipient의 paired L0 graph를 측정 기반 CPU 병렬 작업으로 구성하고, 검토된 `recommend`로 L0/L1/L2 점수를 계산한다. 같은 `PlacementSpec`으로 사후 eligibility를 판단한다. raw score와 raw rank를 보존하고 eligible 최고점 또는 None을 반환한다. CT/mask/anchor는 선택에서 native paste까지 hash와 좌표로 대조한다.

`tools/v22_online_rank_adapter.py`의 RPC/reader와 `RankedLoader`는 선택된 번호를 그대로 사용한다. raw argmax를 다시 계산하지 않는다. None이면 원본이며 CP 여부·donor·candidate·intensity scale·shift의 5회 난수 소비는 보존한다. 새 trainer는 `custom_trainers/nnUNetTrainer_OnlineRankV22.py`이다. frozen GNN 생성은 segmentation Torch RNG를 보존한다. native train/validation GPU 연산과 GNN 추천은 lock을 공유한다.

128개 후보 graph를 디스크 캐시로 만들지 않는다. CP 이벤트마다 필요한 graph를 만들고 점수 계산 후 해제한다. 선택된 raw payload와 수신 CT의 공통 native 준비 결과는 저장된다. 모든 donor×recipient 조합을 미리 만들지 않는다. 다만 장기간 여러 조합이 요청되면 선택 payload도 누적되며, 이 저장량을 무제한으로 안전하다고 주장하지 않는다. 동시 이벤트의 전체 graph 재고는 하나씩 유지하고 각 이벤트 내부의 graph 구성과 GPU batch는 병렬화한다. full-support GNN과 segmentation 공존 시 batch/VRAM 실측은 남아 있다.

## 캐시 용량과 생성 가드

기존 전체 cache의 참조 파일 크기 합은 **20,772,401,500 bytes (19.35 GiB)**이다. graph 14,102개 16.58 GiB, 공유 source 527개 0.54 GiB, donor 준비 527개 2.22 GiB. 논리 파일 크기이며 hardlink가 공유하는 디스크 할당량은 중복될 수 있다. 원본 CT·다른 버전·checkpoint는 포함하지 않는다. 이전 source의 크기 조사이며 최신 학습 캐시로 검증한 것이 아니다.

`tools/v22_cache_storage.py`는 실제 참조 파일의 크기를 읽고 category별 최대 크기 × 목표 전체 개수로 추정한다. 최신 기하의 엄밀한 상한이 아닌 경험적 추정임을 출력한다. 원본 CT 및 checkpoint 용량도 별도다. 추정량과 기존 80 GiB 여유 공간이 없으면 생성 전에 거부하고, 쓰는 동안 기존 writer 여유 공간 검사도 유지한다.

`tools/v222_prepare_optimized.py`는 `--storage-plan` 없이 생성하지 않는다. 계획은 실제 observation index hash·전체 개수·출력 경로와 결속된다. `tools/run_v222_server.py`의 새 전체 준비에는 `--storage-reference-cache`가 필요하며 observations 다음에 저장량 산정 후 cache를 생성한다. 기존 `--cache` / `--resume` 경로는 새 캐시를 만들지 않는다. 기존 cache 파일을 복사하거나 source hash를 바꾸어 재사용하는 기능은 아니다.

```bash
# 기존 캐시는 크기 측정에만 사용한다. 이 두 경로는 실제 서버 경로로 지정한다.
python tools/v22_cache_storage.py --reference-cache "$REFERENCE_CACHE_INDEX" \
  --index "$OBSERVATION_INDEX" --output "$NEW_CACHE_DIRECTORY" \
  --report "$STORAGE_PLAN_JSON"
python -u tools/v222_prepare_optimized.py --index "$OBSERVATION_INDEX" \
  --output "$NEW_CACHE_DIRECTORY" --storage-plan "$STORAGE_PLAN_JSON"
```

현재 전체 cache 생성은 실행하지 않았다. 위 명령은 사용자 실행용 형식이며 이미 준비된 observation과 새 출력 경로가 필요하다.

## 별도 온라인 실행 진입점

```bash
# 완성된 production final ranking checkpoint와 검증된 native 준비가 필요하다.
python tools/v22_online_rank_bank.py --native "$NATIVE_INDEX" \
  --checkpoint "$FINAL_RANKING_CHECKPOINT" --output "$NEW_ONLINE_BANK"
python tools/train_v22_online_rank.py --bank "$NEW_ONLINE_BANK/index.json" \
  --results "$NEW_SEGMENTATION_RESULTS" --workers "$MEASURED_NATIVE_WORKERS" --check-only
```

`--check-only`는 경로·catalog 검사다. 제거하면 foreground에서 실제 250 epoch segmentation 학습이 시작된다. 이번에는 실행하지 않았다. 기존 nnU-Net package를 덮어쓰지 않고 실행 프로세스 안에서 새 trainer를 등록한다. 기존 결과가 있으면 덮어쓰지 않는다. 이 명령 제공은 G3/G4 합격을 뜻하지 않는다.

## 이번 검증과 남은 항목

- 새 DEBUG 테스트 8개 통과: raw 2위 선택, None 원본 유지, 5개 난수 draw, 동점 순서, 잘못된 anchor 거부, proposal의 종양/간 구분 불변, 실제 localhost RPC 전달/실패 전파, 저장량 admission/서버 단계 연결.
- native parity는 synthetic CT에서 **실제 resampling 및 실제 native loader**를 실행했다. 출력 CT는 상대/절대 허용오차 4e-6 이내, segmentation은 exact. 점수는 명시적인 테스트 입력이다. 모델의 임상 성능 결과가 아니다.
- 관련 기존 회귀 56개 통과. 새 테스트와 합해 64개이며 중복 실행은 합산하지 않는다.
- 실제 CT liver_31 / donor liver_73, 전체 모델·기존 DEBUG support8·후보3 진단: 1위 coverage 부족, 2위 종양 overlap 제외 후 3위 선택. 실제 새 selection 계약도 index2를 수락했다. 2,368 voxel raw paste와 주석 변경 시 점수 불변을 확인했다. workers0/1/2/4/8 결과가 같았다. 추가 interior 후보는 DEBUG 대조군이다.
- 실제 CT GNN 진단과 synthetic native 전체 입력 진단은 **분리된 검사**다. 실제 전체 CT → owner builder → RPC → native 최종 입력 전체를 하나의 production 실행으로 검증했다고 주장하지 않는다.
- 로컬 자원: RTX5070Ti 16GB, CPU8/16, 가용 RAM 약45.8GiB. 이번 native 단위 검사는 CPU, 실제 CT 추천은 GPU였다. G3 전체 규모 처리량이나 segmentation 공존 peak 측정 결과는 아니다.
- pair 비균등 기여도(D01), stale L0 reference, 종양 appearance shortcut, 고정 donor 효용 문제는 원래 학습 목적에 남아 있다. 이번 온라인/저장 수정으로 해결됐다고 하지 않는다. 연구 목적 변경 없이 고칠 연결 누락을 먼저 수정했으며, 재가중/CE 삭제/중심 masking을 몰래 적용하지 않았다.
- G3 전체 support/worst-batch/다른 다음 batch 재개/coverage, 실제 전체 CT의 native 통합 및 segmentation 공존(G4), CP 효용(G5)은 미완료다. production cache·GNN40epoch·segmentation250epoch·전체 평가를 실행하지 않았다.

초기 테스트는 Windows 임시 폴더 쓰기 권한에 막혔다. 직접 생성한 테스트 PID와 명령을 확인한 뒤 그 PID만 종료하고, 프로젝트 전용 임시 폴더에서 재실행해 통과했다. 기존 학습이나 원격 세션은 종료하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch 및 병렬화 경로를 확인했다. 새 공존 상황의 최적 batch는 미측정이다.
- [x] 로컬 GPU, CPU, RAM 활용 상태를 확인했다.
- [x] OOM 회피를 위해 모델을 축소하지 않았다. 이번 OOM은 없었다.
- [x] DEBUG 실행과 최종 설정을 분리했다.
- [x] production에 dummy, placeholder, random fallback을 넣지 않았다. synthetic 시험 입력은 명시했다.
- [x] 기존 학습 forward/loss/gradient/optimizer를 보존하고 추천 경로를 실제 loader에 연결했다.
- [x] 실행 범위와 미검증 항목을 기록했다.
- [x] smoke/단위 검사와 전체 학습·평가를 구분했다.
