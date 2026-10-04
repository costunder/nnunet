# v1.4 10mm 범위 축소 — 원본 v1 학습 조건을 맞춘 CUDA 진단

이번 질문은 **원본 v1의 L0 물리 범위를 10mm로 줄이면 source-anchor 순위 학습이 망가지는가**다. v2.2의 관측 P/U 정답이나 L1/L2로 교체하는 실험이 아니다. 기존 native/30mm 결과를 보존하고 **10mm arm만 실행**한다. v1.4는 v1.0에서 범위만 변경한 분기다.

## 기존 짧은 결과의 비교 조건 수정

이전 10mm 2-update 결과는 curriculum epoch1/2와 cosine scheduler를 사용했다. 재사용하는 native DEBUG 결과는 epoch29 loss, 고정 train view29, validation effective view0, scheduler 없음이었다. 따라서 두 결과를 범위만 바꾼 대조라고 제출하면 안 된다.

`tools/verify_v14_learning_replay.py`는 native 실제 CT/CUDA 원시 결과와 fixture의 파일 SHA 및 설정을 검증한 뒤, 10mm에도 다음 조건을 적용한다.

- 같은 실제 CT: train liver_5/liver_6, validation liver_31. 이 세 환자는 **로컬 DEBUG fixture**이며 서버 전체 데이터셋이 아니다.
- 같은 초기 neural hash, 원본 10,434,532 parameter 모델, 3층 L0 / 2층 L1 / 2층 L2, hidden128, heads4, 원본 CNN, 두 view.
- 같은 후보 순서·GT·transform·difficulty/corruption·patient/prototype 입력. 후보8/pool128 유지.
- 같은 seed42, physical sample batch2 / 각 view16 후보 graph, workers4, 원본 AMP·AdamW·clip5·consistency 및 curriculum epoch29 loss, scheduler 없음.
- 첫2 update는 보존한 native2 update와 비교한다. 10mm만 명시적 **성공 optimizer update16회**까지 진행해 작은 batch의 지속적인 학습 여부를 검사한다. production epochs40 설정은 변경하지 않는다.
- CUDA12GiB / RSS32GiB / wall600초의 명시적 진단 자원 한도를 사용한다. 한도 초과는 실패로 보고한다. 모델·샘플·후보를 자동 축소하거나 CPU로 전환하지 않는다.

native 보고서에는 당시 Torch/PyG/CUDA 버전 및 최종 RNG/optimizer 전체 상태가 기록되지 않았다. 따라서 이것은 **기록된 학습 recipe와 초기 상태를 맞춘 비교**이며 bitwise 재현이나 exact resume를 주장하지 않는다. 사용한 원본 v1 source202파일과 실제 CT/fixture/reference bytes를 보존한다.

## 수치와 실패 기록

최종 수치는 `validation/v14_matched_learning_20261004/matched10mm/report.json` 및 `updates.jsonl`, `curve.jsonl`, `attempts.jsonl`을 근거로 읽는다. `matched10mm/executed_runner.py`는 실제 실행 코드다.

| 10mm 고정 train batch | 초기 | 2 update | 16 update |
|---|---:|---:|---:|
| MRR | 0.3500 | 1.0000 | 1.0000 |
| top1 | 0.0000 | 1.0000 | 1.0000 |
| positive−best-other 평균 margin | −0.00739 | +0.02412 | +2.77344 |

complete original loss는 첫 update **3.8925 → 마지막1.0871**로 감소했다. stochastic training loss이므로 매 step 단조 감소했다고 주장하지 않는다. CNN/L0/role·shell pooling/fusion/L1/L2/scalar score의 gradient는 **16회 성공 갱신 모두 finite·양수**, 전체 **1,085/1,085 trainable tensor**가 성공 update의 backward에 연결됐다. score head의 실제 weight 변경도 확인했다.

같은 첫2 update 기준 native는 train MRR0.3125→1.0000 / top1 0→1, 10mm는0.3500→1.0000 / top1 0→1이다. 즉 **작은 고정 batch에서10mm는 원래 순위 목표를 배울 수 있었고,16회까지 순위 분리도 유지·확대됐다.** 학습능력 자체가 끊겼다는 결과는 아니다.

반면 별도 환자는 MRR **0.2500→0.1429**, top1은0으로 유지됐다. 따라서 **전체 품질 유지·일반화·CP 추천 품질은 미검증**이다. 보존한 native의 같은2-update held-out MRR도0.2000→0.1429로 감소했다. native는16-update 결과가 없으므로 10mm16회의 전체 곡선을 native16회와 직접 비교했다고 주장하지 않는다.

실제 sampled train graph의 첫 view 평균은 다음과 같다. 두 view와 모든 관계·edge 규칙은 유지하며 별도 node/edge cap은 넣지 않았다.

| 한 후보 graph | native 원시 기록 | 10mm | 감소 |
|---|---:|---:|---:|
| node 평균 | 13,138.875 | 4,628.000 | 64.8% |
| edge 평균 | 889,816.563 | 373,628.063 | 58.0% |

RTX5070Ti에서 성공 update 중앙값12.410초, peak GPU allocated2.525GiB, process peak RSS4.382GiB, 준비·관측·평가를 포함한 진단 wall315.405초였다. 다른 프로젝트와 공유한 GPU의 DEBUG compute이며 loader/전체 cache 준비/전체 validation/production checkpoint 비용이 포함된 서버 epoch 속도 비교가 아니다. 같은 장비의 동시 native 시간 대조도 실행하지 않았다.

첫 실행 r1은 12회의 optimizer 갱신 후 13번째 backward에서 기존 검사기의 “0 또는 비유한 core gradient” 오류로 중단됐다. 그 검사는 둘을 구분하지 않았으므로 r1만으로 원인을 확정하지 않는다. 두 번째 r2는 추가한 자원 기록에서 WindowsPath를 psutil에 직접 전달해 학습 전에 실패했다. 경로를 문자열로 전달하도록 수정했다. **두 실패의 코드·원시 기록을 각각 보존**했다.

최종 검사기는 gradient zero/NaN/Inf를 구분해 기록한다. 비유한 gradient가 나오면 clipping을 적용하지 않고 **원본 GradScaler의 overflow skip과 scale 감소**, fused AdamW의 실제 optimizer counter 불변을 확인한다. 이 attempt를 성공 update로 세지 않는다. 첫2 native 비교 중 overflow는 비교 조건 불일치로 실패한다. 이후에도 skipped attempt와 성공 update를 별도 보고한다. 전방 loss 비유한값은 즉시 실패한다. finite zero gradient는 기록하며 모든 core gradient가 양수였다고 임의 승인하지 않는다.

최종 실행은 backward attempt17회 중13번째에 실제 AMP overflow가 발생했다. optimizer counter는12로 유지되고 scale은65536→32768로 감소한 뒤16회의 성공 update를 완료했다. `attempts.jsonl`에 비유한 gradient group·tensor명과 skip 확인을 남겼다. 수치 오류를 숨긴 fallback이 아니라 원본 AMP 동작을 관측한 것이다.

`failed_r1`, `failed_r2`, 원본 `native_reference`, 실제 fixture metadata 및 최종 진단은 전달 evidence에 포함한다. CT 파일과 fixture tensor 자체는 Git에 올리지 않는다. 단위 검사는 실제 기록의 metadata/byte 계약을 검사하며 GPU 학습의 대체가 아니다.

최종 metadata 계약13개와 기존 버전·재개 계약7개, 합계 **20개 단위 검사 PASS**다. 원시 JSON/JSONL 결속, 원본CT·fixture·native 기록, 실제 실행 코드 SHA 및 원본snapshot **202/202파일 byte 일치**를 확인했다. `verification.json`은 이 전달 evidence의 파일SHA/크기와 검증 범위를 기록한다. 통합 CUDA 결과는 단위 검사와 별도로 위16회 실제 update에서 나온 것이다.

## 해석과 서버 단계

이 진단은 작은 고정 batch에서 **정답 후보의 순위·margin·loss가 실제로 학습되는지, CNN/L0/L1/L2/score와 optimizer가 연결되는지**를 확인한다. 별도 환자1명의 짧은 결과를 전체 validation21 정확도나 CP 추천 품질로 승격하지 않는다. 기존 native의 같은 짧은 대조도 별도 환자 성능은 개선되지 않았다. 그 사실과 역사적인 v1 전체 학습 성능은 분리한다.

역사적 원본 v1 서버 로그에는 best epoch30 validation MRR0.9869 / top1 0.9804가 기록돼 있다. 이는 보존된 사용자 로그의 수치이며 이번에 전체 모델을 재학습해 독립 재현한 결과가 아니다. 당시 split105/26과 현재 비교 split84/21도 구분한다.

실제 서버 실행은 [전체 10mm 학습 문서](v1_scope_learning_20261004.md)의 `tools/run_v14_scope_training.py`를 사용한다. **전체84 train /21 validation, outer26 제외, seed42, 원본40epoch**다. 성공한 native sample의 후보를 검증·재사용하고, native 실패/누락 요청은 원래 builder로 준비하며 `paired_native=false`를 기록한다. 기존 30mm/native를 다시 실행하지 않는다.

같은 실험 폴더·범위·자원으로 재실행하면 원본 checkpoint의 마지막 완료 epoch부터 이어간다. helper/source/GT가 다른 checkpoint는 거부한다. 물리 GPU와 margin은 CLI로 지정한다. physical batch/worker는 원본 CUDA calibration을 사용하며 로컬 DEBUG batch2를 production 기본값으로 복사하지 않는다. 장기 로컬 학습·128-candidate production CP·nnU-Net은 시작하지 않았다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 사용자가 요청한 물리 범위만 별도 arm에서 바꿨다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 기존 실제 CT fixture의 명시적 DEBUG 실행을 분리했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. native와 같은 physical batch/workers를 사용했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 본 진단에서 모델 축소를 적용하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있는지 실제 CUDA로 검사했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
