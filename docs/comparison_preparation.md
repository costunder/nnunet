# 비교 실험의 CPU 입력 대기 개선

대상은 기존 `selected`, `native`, `native_fixed`, `native_listwise` 10 mm 실험이다. 학습 입력을 준비하는 반복 CPU 계산을 재사용한다. 모델, loss, 후보, 실제 graph view, epoch, physical batch, optimizer와 저장된 진행 상태는 변경하지 않는다.

## 서버에서 확인한 병목

사용자가 제공한 `native_fixed` epoch 2의 완료된 14개 update 기록 기준이다. 서버 파일을 직접 읽거나 서버 프로세스를 조작한 결과가 아니다.

| 측정 | 시간 |
|---|---:|
| 입력 대기 + update | 1,531.92초 |
| 입력 대기 | 1,408.66초 (91.95%) |
| update | 123.26초 |
| forward / backward | 26.73 / 53.35초 |
| CPU → GPU 전송 | 0.31초 |
| 체크포인트 저장 (update에 포함) | 39.32초 |

CPU 입력 생성의 1,497.85초 중 source 준비가 405.49초, 기존 미분류 조립 구간이 909.68초였다. 반면 canonical local graph는 7.61초, sampled views는 26.02초였다. 이 입력 시간은 GPU와 겹칠 수 있으므로 위 전체 시간에 더하지 않는다. 메모리 압력 이벤트는 0이었다. 이는 일반 LRU eviction이나 NFS 지연이 없다는 뜻은 아니다.

미분류 조립 구간에는 상위 graph 생성, 처음 읽는 source/lesion helper, raw identity 검사, patch 조립, collate가 포함된다. 전체 909.68초를 한 함수의 비용이라고 단정하지 않는다. 새 기록은 이 구간을 더 세분한다.

## 적용하는 실행 변경

- **source 준비:** 최초에는 검증된 원본 `choose_source_tumor`와 `prepare_local_source`를 그대로 실행한다. 선택된 종양의 작은 patch mask, 정확한 centroid와 metadata, 준비된 원본 source graph를 저장한다. 다시 읽을 때 full mask를 정확히 복원하고 기존 RAM LRU와 예산으로 관리한다. 전체 CT 크기의 source mask를 파일에 반복 저장하지 않는다.
- **간 통계와 source 방향:** 원본 `_liver_raw` 반환 배열을 환자/region별로, `_principal_axis` 반환값을 source별로 재사용한다. 공식이나 부동소수점 집계 순서를 다시 구현하지 않는다. 다른 병변의 mask는 source 축 캐시에 섞이지 않는다.
- **측정:** 상위 graph, prototype graph, binding, collate를 별도로 기록한다. 내부 helper 시간은 부모 구간과 겹치는 세부 측정으로 표시하고 총시간에 중복 합산하지 않는다.

새 파일은 각 실험이 소유한 `data/source_prepared`와 `data/compact_upper_v1`에 들어간다. 기존 source, checkpoint, canonical cache와 manifest를 덮어쓰거나 삭제하지 않는다. raw SHA, 원본 code/scope, CT shape/spacing, 설정, source identity를 결속하며 손상된 결과를 조용히 새 값으로 대체하지 않는다. 이미 실행 중인 프로세스에는 자동 적용되지 않는다.

## A6000 VRAM과의 관계

서버에서 관측된 training peak는 11.12 GiB다. 남는 VRAM이 있지만 지금 대기의 대부분은 전송 이전의 CPU 계산이다. 전송 0.31초를 줄이는 device prefetch만으로 1,408.66초를 해결할 수 없다. 입력을 제때 공급하는 것이 먼저다.

이 패치는 VRAM을 인위적으로 채우지 않으며 현재 비교 실험의 batch를 중간에 바꾸지 않는다. 입력 대기가 줄어든 뒤 physical batch 또는 activation 저장 정책을 비교하려면 최대 크기 graph를 포함한 실제 CUDA 측정이 필요하다. 관측된 peak 하나만으로 나머지 VRAM 전부를 안전하게 사용할 수 있다고 판단하지 않는다. 학습 중 바뀌는 CNN/GNN embedding을 오래 캐시해 gradient나 가중치 갱신을 건너뛰지도 않는다.

## 검증 범위

검증 결과는 `validation/comparison_preparation/verification.json`에 기록한다. 로컬 상세 자료는 `work/runs/v1.9/preparation/debug-20261008/`에 보관한다. CPU 단위 검사와 실제 CT/CUDA DEBUG 검증을 production 처리량·40 epoch 학습·추천 품질과 구분한다. 실행 결과 없이 서버 epoch 시간이나 가속 배수를 약속하지 않는다.

관련 단위 검사 104개가 통과했다. 실제 CT 세 환자의 보존된 fixture에서 train8 epoch 1/2와 val129를 cold/hot/메모리 eviction/새 provider 재열기의 네 조건으로 비교했다. 총 12개 배치 대조에서 값·dtype·shape·stride·PyG metadata·전역 RNG와 원본 source/준비 graph/helper 값이 같았다. 학습 graph view의 실제 epoch를 고정하지 않았다.

위 입력 일치 검사의 대조군은 보존된 원본 직렬 provider다. 그 검사의 시간에는 이전에 배포한 병렬 view와 국소 mask 추출 효과도 포함되므로, 이번 패치만의 개선 수치로 사용하지 않는다.

추가로 **현재 배포 경로의 병렬 view·국소 mask·메모리 예산을 그대로 적용한 대조군**과 비교했다. 같은 실제 CT train8 epoch 2 배치를 세 번씩 준비하고 실행 순서를 교대했을 때 중앙값은 기존 9.492초, 새 경로 3.176초였다. 입력 준비 시간 약 66.5% 감소다. RAM eviction 후에는 30.490초 → 17.159초였다. val129 한 쌍은 20.920초 → 19.089초로 개선이 작았다. 검증의 후보 graph 생성·sampling까지 모두 없어진 것이 아니다. 총 여섯 쌍의 입력·graph metadata·RNG가 같았다. 로컬 Windows SSD의 세 환자 DEBUG fixture이며 서버 NFS, 네 실험 동시 실행 또는 epoch 전체 시간의 가속 배수가 아니다. 새 경로의 disk cache는 앞선 검증에서 생성한 정확한 완료 결과를 사용했다. 새 source를 처음 만들 때는 원래 계산과 저장 비용이 여전히 든다.

전체 원본 모델 10,434,532 parameters로 실제 CUDA optimizer update 두 번과 129후보 validation 세 번을 수행했다. 1,085개 parameter tensor의 gradient가 모두 유한했다. 완료 checkpoint 직접 재개와 복제된 완료 실험의 실제 재개 launcher는 각각 추가 update 0개였고 원본 checkpoint·cache·17개 결속 helper는 보존됐다. RTX 5070 Ti에서 관측된 DEBUG peak는 CUDA 2.199 GiB, RSS 16.37 GiB다. 서버 A6000의 최대 graph나 production batch 한계를 측정한 값이 아니다. 상세 입력/학습/성능 기록을 검증 폴더에 함께 보관한다.

세 source의 새 준비 캐시는 총 12,895,896 bytes, 간 통계/축 캐시는 7,587 bytes였다. 이는 전체 서버 cohort 저장량이 아니다. source payload에 전체 CT mask가 포함되지 않는지 실제 파일을 읽어 확인했다.

## 완성된 입력 재사용으로의 복구

이전 source·간 통계 캐시만으로는 v1.8/v1.9의 입력 경로가 복구되지 않았다. 원래 v1은 완성된 canonical sample을 mmap으로 읽고 epoch별 view를 만들었지만, 확장 경로는 각 sample에서 CT/source/후보 metadata와 상위 graph를 다시 조립했다. `workers=16`도 완전한 sample 16개를 준비하는 설정이 아니라 주로 sample 내부 후보 작업을 병렬화하는 값이었다. 10 mm 공간 범위를 유지해도 반복 조립 비용이 추가된 실행 경로의 문제다.

서버의 완료된 epoch 1은 train/load/save 14,811.6초, validation 5,491.2초, 합계 20,302.8초였다. 학습 부분만 약 4시간 7분이므로 8개에서 129개로 늘어난 검증만으로 이 지연을 설명할 수 없다. 해당 시간은 batch에서 누적한 실제 활성 시간이며 통상적인 중단·재개 사이의 대기 시간을 더한 값이 아니다.

새 `sample_layout`은 원본 canonical sample의 작은 상위 graph와 metadata만 저장하고, 큰 local graph 및 CT patch는 **기존 canonical 파일을 참조**한다. 같은 후보 구성을 다시 요청하면 CT decode, source 준비, 원본 상위 graph 조립을 건너뛰고 원본 epoch별 두 sampled view를 생성한다. RAM hit마다 NFS 파일을 다시 검사하지 않으며, 기존 resident LRU에서 사라진 데이터는 다시 읽을 때 SHA와 binding을 검증한다. 원본 CT는 프로세스에서 환자별 최초 사용 시 검증한다.

완성된 layout이 있는 배치는 RSS/resident 예산 안에서 여러 배치를 미리 준비한다. 최초 입력의 RSS 증가와 실제 tensor storage를 측정해 동시 준비 수를 정하고, 공유 candidate pool 하나를 사용해 batch마다 worker pool을 중첩 생성하지 않는다. candidate pool은 저장된 worker 수를 그대로 유지하고, 대기·조립을 맡는 batch 조정 thread는 별도로 제한한다. 대부분 candidate 작업을 기다리는 조정 thread 때문에 계산 worker 수를 빼지 않는다. PyTorch/NumPy 내부 thread 설정도 그대로다. 전체 OS thread 수를 worker 설정과 동일하다고 보고하지 않는다. 출력 순서·epoch seed·physical batch·optimizer update·resume cursor는 유지한다. cold 입력은 기존 조립 경로를 사용하며 다른 CPU batch와 동시에 조립하지 않는다. 선택된 동시 준비 수와 대기는 각 arm의 `prefetch.jsonl`에 기록한다. 메모리 추정은 최대 크기 입력에 대한 보장이 아니므로 기존 hard budget 검사도 유지한다.

`selected`/`native_fixed`의 반복 후보와 고정 val129는 layout을 재사용한다. `native`/`native_listwise`는 epoch마다 7개 U의 조합이 바뀌므로 **처음 보는 조합의 상위 조립은 남는다.** 원본 local graph/source 캐시는 재사용하지만, 아직 없는 layout을 cache hit라고 표시하지 않는다. view sampling·collate·GPU 학습·checkpoint 저장도 계속 수행한다. 학습 중 변하는 embedding을 저장해 역전파를 생략하지 않는다.

이 실행 변경의 실제 입력·CUDA·resume 검증은 `validation/comparison_loading/`에 별도로 기록한다. 앞 절의 수치는 이전 source/upper 캐시 검증이며 새 layout의 성능으로 재사용하지 않는다. 로컬 DEBUG 시간과 서버 40 epoch의 처리량은 구분한다.

실제 CT 세 환자에서 네 군의 train8 epoch 1/2 및 val129 요청을 확인했다. 완전히 같은 source·후보 순서·epoch view 요청만 합치면 서로 다른 요청은 6개이며, cold/hot/LRU eviction/새 provider 재열기의 네 조건에서 총 24개 입력이 값·dtype·shape·stride·metadata·RNG까지 일치했다. 재사용 18개 요청에서는 CT/source/상위 graph/원본 inference builder 진입점을 실패하도록 바꾼 검사도 통과했다. 새 layout 7개는 합계 862,106 bytes이며 큰 graph와 dense patch 중복 파일은 없었다.

**완전히 RAM에 올라온 직전 배포 `9c36b86` 대조군**과의 train8 입력 중앙값은 3.476초 → 3.167초였다. val129 한 쌍은 12.948초 → 12.993초로 개선되지 않았다. 이는 기존 source/upper 캐시가 이미 효과를 내는 상태이며, 새 layout이 모든 정상 상태 계산을 크게 가속한다는 근거가 아니다. 원본 view 생성은 남는다. 메모리에서 제거하거나 새 provider로 다시 열어도 CT/source 재조립이 발생하지 않는다는 점과, 완료된 입력을 여러 배치 준비하는 실행 연결을 별도로 검증했다. 서버 NFS의 실제 시간·최대 크기 graph·네 군 동시 실행의 처리량은 미측정이다.

최종 공유 candidate pool은 측정된 worker 4개를 그대로 유지했다. 실제 CT physical batch 2의 배치 세 개를 순서대로 준비하는 방식과 queue 방식으로 세 쌍 교대 측정했다. 총 18개 배치의 입력·순서·RNG가 같았고, queue에서는 두 배치 준비가 실제로 겹쳤다. 세 배치의 CPU 준비 중앙값은 9.330초와 8.944초로 차이가 작았다. 해시 검사는 시간 측정 밖에서 수행했다. 이는 CUDA 소비 작업 없는 로컬 CPU 측정이며 서버 epoch 가속 배수로 환산하지 않는다.

최초 입력 검사 직후에는 queue 상태 필드를 진행률의 시간 필드에 전달하는 오류로 첫 CUDA forward 전에 중단됐다. 그 실패를 보존하고 전달 형식을 원래 두 시간 필드로 고쳤다. 이후 candidate worker를 배치 조정 thread 수만큼 빼던 배분도 공유 pool 하나의 원래 worker 수로 수정했다. 입력 구현이 바뀌지 않은 24개 대조는 당시 코드 SHA와 함께 보존하고, 변경된 staging 경로는 최종 코드에서 다시 검사했다. `verification.json`은 각각의 코드 결속과 최종 CUDA 실행을 구분한다.

최종 통합 단위 검사 143개와 Python 3.10 구문 검사가 통과했다. 실제 CUDA에서 원본 10,434,532 parameter 모델로 physical source batch 2·worker 4·optimizer update 2번·full129 validation 3번을 실행했다. 1,085개 parameter tensor의 gradient가 모두 유한했고, 완료 checkpoint의 직접 재개와 실제 controller 재개에서 추가 update는 각각 0개였다. 원본 실험·cache·17개 결속 helper를 보존했다. 로컬 RTX 5070 Ti의 기록된 training peak는 CUDA 2.199 GiB, RSS 15.375 GiB였다. 이는 DEBUG 실행 검증이며 production 40 epoch 학습, 전체 cohort 평가, A6000 서버 처리량 및 추천 품질 검증은 수행하지 않았다.

## GPU 실행 설정의 실제 측정

CPU 입력 재사용에 더해 기존 모델의 `dense_batch_size`, `checkpoint_dense_encoder`, `checkpoint_local_blocks` 세 실행 속성을 측정한다. source physical batch와 8개 학습 후보, 129개 검증 후보, 모델 구성, loss, Adam, epoch와 cursor는 유지한다. CNN patch chunk를 늘리는 것은 optimizer batch를 늘리는 것이 아니다.

같은 실제 batch에서 원래 설정과 activation 보관 조합을 비교하고, 통과한 조합의 CNN chunk를 실제 patch 개수까지 늘려 측정한다. warmup을 제외한 forward/backward 시간, peak VRAM, 점수·loss·전체 gradient 차이를 기록한다. optimizer step은 수행하지 않으며 모델·gradient·RNG·buffer·모드를 복구한 후 실제 학습을 진행한다. 측정 중에는 GPU trial 진행이 표시된다.

선택 조건에는 기존 CUDA 예산, 가용 메모리, 여유 공간과 optimizer 임시 메모리가 포함된다. 작은 batch 측정을 전체 cohort의 최대 메모리 보장으로 쓰지 않는다. 측정 범위를 넘는 입력은 원래 실행 설정을 사용하고, workload가 크게 증가하면 해당 실제 batch를 다시 측정한다. 최적화 설정에서 forward/backward OOM이 발생하면 optimizer 갱신 전에 참조를 정리하고 같은 입력·RNG로 원래 설정을 재시도한다. 후보나 graph를 버리지 않는다. 원래 설정도 실패하면 오류를 보존한다.

각 arm의 `gpu_execution.jsonl`에는 측정과 선택 근거가, `update_timing.jsonl`에는 실제 적용한 설정과 교정 시간이 남는다. `report_comparison_runtime.py`는 최근 설정과 forward/backward 측정값을 함께 표시한다. 교정 시간은 step 시간에 포함되며, 교정의 가속 배수는 epoch 전체 가속 배수가 아니다. validation은 기존 chunk와 실행 설정을 유지한다. 완료 checkpoint를 재개하면 GPU 교정과 추가 update를 하지 않는다.

실제 로컬 CUDA 검증과 서버 A6000 측정은 구분한다. 서버의 새 epoch 시간은 사용자가 해당 Git 실행 경로로 재개한 후의 기록으로 확인한다.

최종 실제 CT/CUDA DEBUG 검증은 `validation/comparison_gpu/verification.json`과 원본 `report.json`에 기록했다. 전체 모델 10,434,532 parameters, physical source batch 2, workers 4, train8/two views와 val129를 유지했다. RTX 5070 Ti에서 같은 checkpoint를 각각 복제한 실제 AMP/GradScaler/AdamW update 중앙값은 **9.549초 → 6.597초**, peak CUDA는 **2.200 → 7.079 GiB**였다. activation을 보관해 backward 재계산을 줄인 결과이며 약 1.45배는 GPU 계산 처리량이다. CPU 준비·checkpoint·epoch 전체의 가속 배수가 아니다. CNN chunk 4/8/16도 측정했으며 이 checkpoint에서는 chunk 4가 가장 빨랐다. 서버에 이 값을 고정하지 않고 서버의 실제 batch와 저장된 CUDA 예산으로 다시 측정한다.

교정 자체는 139.625초였고, forward/backward 130.336초와 상태 보존·검사 등 9.289초를 구분했다. 처음 요청한 batch나 크게 증가한 workload에서 측정 시간이 들 수 있다. 이 시간은 실제 wall-time에 남기지만 매 update마다 반복되는 비용처럼 ETA에 곱하지 않는다. 점수와 loss뿐 아니라 1,085개 gradient tensor와 전체 gradient vector 및 RNG 소비를 검사한다. 원래 CUDA 실행도 반복 간 작은 수치 차이가 있어 원본 반복 오차와 명시한 AMP 허용값을 함께 기록하며 bitwise 동일 학습 궤적을 주장하지 않는다. 교정은 unscaled loss backward이며 실제 scaled AdamW update 네 번을 별도로 확인했다.

실제 실행 경로에서 2회 update·3회 full129 validation, 1회 update 후 pause/resume와 완료 재개의 추가 update 0회를 확인했다. 최적화 forward에 OOM을 주입한 DEBUG 검사에서는 같은 batch·모델·RNG로 원래 실행 설정을 재시도하고 checkpoint update/attempt가 각각 1임을 확인했다. 이는 실제 하드웨어 OOM 측정이 아니라 오류 복구 검사다. 원본 설정·source·cache inventory·결속 helper를 보존했다. 최종 166개 단위 검사와 변경 Python 13개 파일의 Python 3.10 구문 검사를 통과했다. CUDA 실행 후 ETA 표시와 verifier 설명 문구만 수정한 내역은 `execution_note.json`에 검증 당시와 배포 코드 SHA를 나눠 기록했다.

A6000 최대 graph, NFS에서 네 군 동시 실행, production 40 epoch 처리량과 추천 품질은 아직 측정하지 않았다. 기존 CPU 입력 재사용 수정도 함께 적용되지만, 회전 후보의 최초 조립과 view sampling까지 없어진 것은 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch와 병렬화 가능성을 검토하고 비교 실험의 기존 측정값을 유지했다.
- [x] 서버 제공 GPU·CPU·RAM 기록과 로컬 자원을 확인했다.
- [x] 모델 축소보다 측정된 CPU 입력 병목을 먼저 조사했다.
- [x] DEBUG 검증과 production 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 학습 경로에 사용하지 않았다.
- [x] 변경 입력의 forward·loss·gradient·optimizer 연결을 실제 CUDA로 검사했다.
- [x] 실행 변경과 유지되는 설정을 보고했다.
- [x] 단위·smoke 결과와 전체 학습·전체 평가 미수행을 검증 기록에 구분했다.
