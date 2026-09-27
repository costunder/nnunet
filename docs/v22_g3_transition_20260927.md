# v2.2 독립 검토 종료와 G3 준비

## 현재 판정

사용자가 제공한 `ea702fd2b7b78faa855ff62a0f2488045a4d96da` 독립 검토 텍스트 전체를 읽었다. C01과 이전 B01–B03은 해당 검토 범위에서 해결됐고 새 확정 버그는 재현되지 않았다. 이번에는 학습/runtime/모델/graph/추천 코드를 수정하지 않는다. **검증할 구현은 ea702fd의 core source + runtime hash로 고정한다.** 이후 문서 커밋은 구현 revision 변경과 구분한다.

검토자의 39 passed/1 skipped/3 deselected, 21개 검사 묶음, 48개 하위 변조 등은 **검토자가 보고한 독립 검사**이며 로컬에서 새로 실행한 검사로 합산하지 않는다. 별도 sandbox ZIP/MD는 제공된 텍스트에 링크만 있어 읽지 않았다. 기존 로컬 86회귀/GPU DEBUG 결과는 C01 기록에 있다.

## 이번에 실제 확인한 준비 상태

`work` 아래 `index*.json`을 조사해 source identity/설정/완전성/DEBUG 여부를 비교했다. graph tensor 전체 hash나 원격 서버 cache를 읽은 검사는 아니다.

| 로컬 자료 | 관측 수 | 판단 |
|---|---:|---|
| 이전 paired production cache의 여러 index | 14,102 (inner train 11,279 / inner val 2,823) | 현재 source와 불일치. 최신 index도 6개 source 차이. G3에 그대로 사용 불가 |
| 최신 content-bound DEBUG cache | 10 (train 8 / val 2) | core provenance 일치, DEBUG/불완전 cohort. G3 대체 불가 |
| 이전 PPR/A*/raw-CT 및 observation metadata | 각각 index에 기록 | 다른 format/source 또는 graph 없는 metadata. 현재 G3 graph cache 아님 |

**조사한 로컬 index 중 현재 구현과 일치하는 전체 production graph cache는 없었다.** 전체 index가 여러 개 있다고 여러 독립 데이터셋이 있는 것으로 합산하지 않았다.

source 차이는 앞선 `placement.py`, `record_binding.py`, `v1_local.py` 등의 변경이다. **C01의 hash 추가 때문에 geometry를 다시 만드는 것이 아니다.** 이미 최신 source와 content binding으로 생성한 전체 cache가 서버에 있다면 실제 검증 후 재사용할 수 있다. 이전 cache에 새 source hash만 붙여서는 안 된다. 로컬 최신 DEBUG 캐시는 이번에도 보존했다.

자원 확인: RTX 5070 Ti 16GB, CPU 8 physical/16 logical, 가용 RAM 약 46.6GB, 여유 디스크 약 1.63TB. 이 수치는 로컬이며 A100 MIG 서버의 실제 할당/가용 메모리는 서버에서 다시 확인한다. 원격 실행이나 전체 cache 재생성, G3 GPU 검증, 전체 학습은 이번에 시작하지 않았다.

## 서버의 첫 단계: 고정 코드와 전체 cache 준비

기존 사용자 지시인 **로컬 검증 → 코드만 Git → 서버 원본 CT로 준비/실행**을 유지한다. 아래는 G3 입력 준비 명령이다. 기존 학습 checkout과 output을 덮어쓰지 않고, 이미 할당된 GPU 환경을 유지한다. 새 경로가 이미 존재하면 실패하므로 재실행 때 기존 결과를 삭제하지 않는다. 모델/graph/전체 관측을 축소하거나 예전 checkpoint를 v5로 변환하지 않는다.

```bash
conda activate nnunet
G3_CODE=/home/aicompetition06/Medical/HierCP-g3-ea702fd
git clone --no-checkout https://github.com/costunder/nnunet.git "$G3_CODE" &&
git -C "$G3_CODE" checkout --detach ea702fd2b7b78faa855ff62a0f2488045a4d96da &&
cd "$G3_CODE"
```

위 clone/checkout 성공 후 다음 준비를 실행한다. 할당된 MIG가 한 개 노출돼 있는지 검사하며, 옛 MIG UUID를 강제로 다시 지정하지 않는다. 검사가 실패하면 GPU 환경 확인이 먼저다.

```bash
python -c 'import os,torch; print("CUDA_VISIBLE_DEVICES:",os.environ.get("CUDA_VISIBLE_DEVICES")); assert torch.cuda.is_available(), "Allocated CUDA required"; assert torch.cuda.device_count()==1, "Verify the allocated single MIG/GPU environment"; p=torch.cuda.get_device_properties(0); print(p.name,p.total_memory/1024**3,"GiB")' &&
python -c 'from pathlib import Path; Path("work/g3_ea702fd_prepare").mkdir(parents=True,exist_ok=False)' &&
python -u tools/v1_server.py resources --output work/g3_ea702fd_prepare/resources.json &&
python -u tools/v1_server.py observations \
  --medical-root /home/aicompetition06/Medical \
  --output work/g3_ea702fd_prepare/observations &&
python -u tools/v222_prepare_optimized.py \
  --index work/g3_ea702fd_prepare/observations/index.json \
  --output work/g3_ea702fd_prepare/paired_cache
```

명령은 foreground에서 각 단계의 progress를 출력한다. 131개 원본 case, split, 전체 관측/공여자 coverage를 기존 production 함수로 준비한다. worker는 기존 측정 기반 scheduler를 사용한다. 위 준비 명령의 GPU 사용률이 낮은 것은 CT/graph 준비가 주로 CPU/I/O 작업이기 때문일 수 있으며, GPU 학습 진행으로 보고하지 않는다. 이 단계는 **G3 통과나 학습 완료가 아니다**.

## G3의 실제 합격 기준

| 검사 | 필요한 증거 |
|---|---|
| 전체 입력 | current-source production cache, 원본 cohort/split 및 모든 record의 누락·중복 없음. 관측 수가 이전 14,102와 다르면 원인을 확인 |
| Full support | 전체 inner-train L0 memory 생성. query recipient/donor 양쪽 제외 후 실제 남은 모든 support 사용. DEBUG support 6/8 대체 금지 |
| Batch 크기 | 예정 physical/effective batch와 accumulation을 명시. 최신 full-support calibration. 이전 physical32 또는 DEBUG batch2를 승인값으로 가져오지 않음 |
| Worst profiles | 실제 bucket batch 중 live node/edge 최대, same-case reference 최대, support/L2 최대, donor 다양성/CPU materialization 부담 등 서로 다른 사례 확인. 단일 group edge 합만으로 최악이라고 단정하지 않음 |
| 메모리·처리량 | Adam이 생성된 이후 forward/backward/update + durable CPU snapshot/쓰기, producer/pinning을 포함한 peak VRAM/RAM·I/O·step time·처리량 |
| 진짜 episode 재개 | 같은 그룹의 서로 다른 다음 batch가 있는 중간 지점. 연속 실행과 재개의 loss/L0·L1·L2 gradient/model/Adam/RNG 비교. 동일 batch 반복 probe로 대체하지 않음 |
| 전체 coverage | 정상 epoch의 모든 query, validation 전체, selected-best final support 전체와 중단재개 결과 확인 |

기존 server runner의 `profile_DEBUG`는 support 6개인 명시적 smoke이므로 G3가 아니다. 기존 production training은 전체 memory를 사용하지만, 여러 worst-profile·실제 다른 next-batch episode의 비교 증거를 자동 완성하지는 않는다. 이 G3 측정 절차/실행 결과는 **아직 완료되지 않았다**. 준비 완료 후 source/runtime을 유지한 별도 진단으로 기록해야 한다. 40epoch 최종 실험 설정을 작은 값으로 덮어쓰지 않는다.

D01 pair 재가중, D02 중심 CT/CE/donor schedule 변경은 별도 연구 실험이다. 이번 baseline에 섞지 않는다. G3 이후 G4 native selection/PlacementSpec/RNG/최종 nnU-Net 입력 연결, G5 동일 조건 no-CP/valid-random-CP/proposed-CP 비교로 진행한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch와 병렬화 계약을 확인했다. 이번에 새 full-scale batch 측정은 하지 않았다.
- [x] 로컬 GPU, CPU, 가용 RAM 및 디스크를 확인했다. 서버 자원은 미측정이다.
- [x] OOM 회피를 이유로 축소하지 않았다. 이번 GPU 학습/OOM 실행은 없다.
- [x] 기존 DEBUG와 앞으로의 G3 전체 규모 검증을 구분했다.
- [x] dummy, placeholder, random fallback을 추가하지 않았다.
- [x] forward/loss/gradient/optimizer 코드는 변경하지 않았다. 기존 검증 범위를 새 전체 규모 결과로 과장하지 않았다.
- [x] 고정 구현과 다음 단계/제한을 명확히 보고했다.
- [x] 이번 작업은 검토 종료·readiness 조사이며 전체 학습/평가를 실행하지 않았다고 명시했다.
