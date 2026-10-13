# Basic CP 7개 독립 실험 실행 계약 — 2026-10-14

| 실제 GPU | 요청한 단일 변경 | CP 위치 후보/source | 성공 CP event의 병변 수 | image blending |
|---|---|---:|---:|---|
| 0 | 병변 5개 | 128 | 5 | hard |
| 1 | 병변 7개 | 128 | 7 | hard |
| 2 | 병변 10개 | 128 | 10 | hard |
| 3 | 후보 512개 | 512 | 1 | hard |
| 4 | 후보 1024개 | 1024 | 1 | hard |
| 5 | 후보 2048개 | 2048 | 1 | hard |
| 6 | donor/recipient 50%/50% | 128 | 1 | constant alpha0.5 |

전체105명 train/26명 validation, 자신의 병변 source642개/CP 가능한81명,
seed42, patch128³, physical/effective batch2, accumulation1, CP gate0.5,
250epoch와 원래 nnU-Net 구성·optimizer·augmentation을 보존한다.
blending은 jitter 이후 병변 mask의 영상 값에만 적용하고 label은 hard다.
후보 확대에서도 기존128개와 검증된256개 prefix 및 실제128개 GNN 점수를
보존하되 위치 선택에는 GNN 점수를 사용하지 않는다.

## 사용자가 승인한 불가능한 첫 위치의 처리

승인 원문: **“허용: 불가능한 첫 위치만 기존 128개 안에서 재추첨”**.

642×128개 위치의 실제 segmentation과 fullN 비중첩 geometry를 전부 검사했다.
5/7개 조건에서는 각각1개, 10개 조건에서는10개의 첫 위치가 불가능했다.
가능한 원래 첫 위치는 그대로 쓰고, 불가능할 때만 동일 환자·동일 source의
원래128개 중 fullN 가능한 첫 위치에서 균등 재추첨한다. source 교체,
위치 추가, CP 생략, 병변 수 축소는 없다. 별도 token RNG를 사용해 기존
gate/source/intensity RNG 순서를 유지한다. 원래 primary 선택 분포가
조건부로 바뀌는 정책은 preflight·native 요청·epoch telemetry·결과 설정에
명시한다. 모든 full source patch를 paste한 뒤 기존 spatial augmentation이
적용되므로 변형 이후 모든 병변이 완전히 잔존한다는 주장은 하지 않는다.

## 공유 후보 준비와 계산 자원

후보는 전체 source의512→1024→2048 순서로 준비하고, 해당642개 source가
모두 완료된 prefix만 공개해 각 GPU의250epoch 학습으로 넘긴다.
source4개씩 묶는 것은 전체642개를 분배하는 scheduling 단위다.
기존 complete256과 이미 공개된 부분 NPZ/receipt를 바이트 그대로 재사용한다.
raw CT 전처리·region 생성·GNN 채점은 다시 하지 않는다.

측정: GPU7개 A6000, CPU96, host RAM 약1008GiB. 초기4worker의 peak RSS는
약35GiB였다. 공유 준비24worker는64개 CPU affinity [24..47,56..95]에서
동시에 실행하며288GiB RSS 예산을 적용한다. native 학습은 GPU별8CPU와
96GiB 예산을 유지한다. 전체 admission은704GiB다. CPU affinity는 실제
사용 가능한 host CPU 기준이며 PBS의 독점 CPU/GPU 예약이라는 주장은 없다.

원래 full graph를 그대로 생성하면서 source에 불변인 geometry field만
512MiB/worker 이하 LRU로 재사용한다. 캐시 eviction은 계산만 다시 수행하며
node/edge/source/후보 수를 줄이지 않는다. 실제 후보20개·ROI 크기2종의
860개 배열(158,470,494bytes)이 원래 full graph와 모두 동일했다.
실측1.31~1.44배 개선이다. 입력·코드·PID·UID·PBS·메모리 전체 검사는
5초 이내 주기와 모든 source/prefix 공개 전에 실행하며, 요청 identity와
실행 identity 및 양쪽 만료 clock은 후보마다 확인한다.

24worker 초기 peak RSS는 약275GiB, 이후 steady는 약150GiB로 측정됐다.
초기 peak 동안 기존 가드가 각 arm에서 `available + 자기 RSS >=704GiB`를
따로 요구해 다른 arm을 중단시킨 오류를 수정한다. 새 immutable registry는
7개 controller의 실제 PID·UID·birth·argv와 request SHA를 고정하고,
중복 PID를 제거한 **전체 group RSS + available >=704GiB**를 검사한다.
GPU0/1/2/6는 각각96GiB, GPU3/4/5는 공유 합288GiB, group은 합672GiB
RSS 상한을 적용한다. 후보 native 학습과 CPU 준비가 겹쳐도 이 합산
예산을 지켜야 한다. 기존 실패 로그와 checkpoint는 삭제하지 않는다.

## 완료·평가·표시 계약

whole epoch checkpoint를 매 epoch atomic 저장한다. full250 학습 후
자기 checkpoint_best.pth로 전체26명 native 예측/평가, 원래 historical
병변/크기별/통계 평가, 고정 Result schema1의 순서로 자동 실행한다.
checkpoint_final.pth는250epoch 완료 증거에만 사용한다.
원래 native console 및 UI2 양식을 유지한다.

재개 감사에서 GPU0/1/2의 완료10epoch와 GPU6의 완료24epoch BEST가
8개 logging 배열·optimizer·scaler를 보존함을 확인했다. 기존 latest는
native `save_checkpoint`의 +1을 postincrement callback에서 중복 적용한
metadata 오류가 있어 그대로 보존하고, 같은 완료 epoch의 full BEST를
새 namespace의 latest로 바이트 그대로 복사해 이어간다. 새 postincrement
저장 helper는 실제 native 함수의 고유 epoch serialization 값만 private
clone에서 수정하며, 원래 BEST/final 저장과 trainer epoch는 변경하지 않는다.
실제 logger는 MetaLogger이므로 내부 속성을 가정하지 않고 public
`get_checkpoint()`로8개 배열을 검증한다. 실제 MetaLogger와 실제 native
saver의 별도 CPU DEBUG 직렬화에서 epoch 값만 달라지고 나머지 상태가
동일함을 확인했다. 이 DEBUG 결과는 학습 성능 결과가 아니다.

2026-10-14 06:08 당시 GPU0/1/2는 실제250epoch 학습을 시작했고,
각각3epoch를 완료했다. GPU6도 실제 학습 중이었다. 후보3개는 공유
CPU 준비 단계로 전체 학습/평가 완료가 아니다. CPU 검사와 별도 CUDA
B2/B4 gradient 검사 통과를 최종 결과로 제시하지 않는다.
현재 host lifetime은13:41:03 KST까지다. 만료 시 전체 규모를 줄이지 않고
실제 checkpoint와 완성 후보를 보존해 후속 할당에서 이어야 한다.
새 결과 저장 위치는 tmpfs이며 durable persistence를 보장하지 않는다.

## 종료 경합과 완료 epoch 보존

GPU0의 완료12epoch latest는8개 logger 길이와 optimizer/scaler가 일치한다.
다른 GPU의 DataLoader가 정상 종료하면서 마지막 argv가 비는 순간을
기존 가드가 오류로 판정했다. 새 v2 가드는 PID·birth·UID가 그대로이고
실제 Linux PF_EXITING이 확인된 경우에만 실제 RSS를 포함해 관찰을 마친다.
live empty argv나 PID 재사용은 여전히 거부한다. 이전 ledger의 정확한 SHA와
해당 row를 고정해 건강한 peer controller를 변경 없이 승계한다. CPU13개
검사 통과이며, 실제 CUDA 재개 확인은 별도 실행 증거로 기록한다.

GPU5의 기존 후보 준비 코드는 immutable 상태로 유지한다. 별도 보조 CLI는
나중에 생성되는 latest의 완료 N epoch를8개 logger로 확인하고, epoch header만
수정한 operational rolling cache를 만든다. 모델·optimizer·scaler·logger·BEST
선택 상태와 serialized device tag를 다시 검증하며 원본 latest/BEST/final은
변경하지 않는다. 이 보조 CLI의 CPU7개 검사는 학습 완료 증거가 아니다.

## 1024 준비 단계의 RAM 복구

512 전체642개 bank 공개 후 GPU3 학습과1024 후보 준비가 겹치면서 기존
공유288GiB 가드가 중단했다. host 전체 RAM은 약1008GiB이며 실패 후
available 약779GiB를 측정했고,800GiB보다 작은 inherited cgroup 제한은
관찰되지 않았다. 기존 환자의 EDT를 캡처한 patient_pool/patient_cache와
이전 source builder가 다음 입력 생성 때까지 유지되는 참조를 해제한다.
정적 patient organ 배열은 재사용하며 모든 source/graph 배열은 그대로다.

새 v3 자원 가드는 candidate GPU3/4/5 합384GiB, 나머지4개 native arm은
각96GiB, group RSS768GiB와 reserve32GiB를 합친800GiB admission이다.
원래24 worker·64CPU, native8CPU·physical/effective B2,128³,250epoch,
전체105/26명과642개 source를 유지한다. 이 v3는 자원 가드 버전이며
GNN/혈관 그래프의 연구 버전을 뜻하지 않는다. 기존 code/request/bank와
checkpoint는 보존하고 새 실행 폴더를 쓴다. 완료512 bank와1024 부분
항목은 원래 파일 SHA·receipt를 확인해 바이트 그대로 재사용한다.

24 worker spawn 직후 기존 exact-affinity 관측이 세 arm을 중단했다.
당시 PID/actual mask 기록이 없어 원인을 단정하지 않는다. 현재 모든
worker의64CPU가 확인됐고, 새 observation revision4는 같은 PID·birth·UID·argv를
유지한 채 최대300ms 내 다시 읽는다. 지속 위반은 실제/예상 mask와 PID를
기록해 실패하며, 실제 PF_EXITING task의 RSS는 계속 계상한다.

## 최종 자원 관측과 원본 BEST 보존

observation revision5는 사라지는 `/proc/PID/stat`의 정확한 ESRCH만
`psutil.NoSuchProcess`로 연결한다. 권한 오류나 다른 I/O 오류를 숨기지 않는다.
native와 공유24worker 모두 기존300ms CPU 재관측 및384/800GiB 계약을
유지한다. 이 변경의187개 CPU 회귀 검사는 CUDA 학습 완료 증거가 아니다.
다른 계정의 CUDA 점유로 실패한 경우는 관측 오류로 처리하지 않는다.
실제 빈 GPU를 기다린 뒤 해당 실험의 whole latest로 재개하며 다른 작업은
종료하지 않는다. 새 namespace의 code/request SHA와 원래 bank를 고정한다.

`/var/tmp` 여유는 약9.57GiB이므로7개 full latest/BEST 쌍(약10.68GiB)을
보존할 수 없다. 별도8GiB/최소 여유1GiB cache에 각 arm의 **원본 full
BEST 하나**를 byte 그대로 보존한다. 모델·optimizer·scaler·8개 logger를
검증하고 모델만 추출하거나 epoch를 바꾸지 않는다. 살아 있는 학습은
whole latest로 계속된다. tmpfs 전체가 소실되면 이 cache의 BEST epoch부터
250까지 재개하므로 BEST 이후 epoch를 다시 수행할 수 있다. 실제 latest
epoch와 저장된 BEST resume epoch 및 전체26명 평가 완료 여부를 구분한다.

전체642 source의 원래128 후보, 원래256 prefix, 공개된512/1024/2048
항목 및 GPU3 native가 사용하는 별도의 원래512 index SHA를 보존한다.
이 cache의 데이터 보존 예산은 모델·source·후보 규모의 축소가 아니다.
보존 프로세스는 CPU82/83을 공유하고 CUDA를 사용하지 않으며 원본
결과·코드·bank를 수정하지 않는다. 실제250 완료와 원래 BEST26 검증이
모두 확인된 뒤에만 평가 완료 역할을 기록한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
