# v1 strict-nested r3 검토 후속 기록

사용자가 전달한 `84e36138-7ad6-4bf3-8a18-b63059b1b15b/붙여넣은 텍스트.txt` 전체를 읽고 현재 runner, sampling contract, 원시 CUDA 보고서를 대조했다. 독립 검토의 외부 manifest 82개와 보존 source 202개 확인, 외부 CPU 검사와 작성자의 실제 CT/CUDA 검사를 서로 합산하지 않는다.

## 판정과 유지할 코드

r3 검토는 새 P0/P1 구현 오류를 찾지 않았으며 sampler를 다시 바꾸지 말라고 권고했다. 실제 train/validation/generate/spawned worker 연결, 동일 canonical preparation 및 measured physical batch/worker 결속, 잘못된 checkpoint 거부와 full21 evaluator 계약은 유지한다.

`nested_graph_size.py`, `graph_size.py`, `graph_flow_audit.py`, `sampling_runtime.py`, `sampling_entry.py`, `experiment.py`, `contracts.py`와 실제 CT checker를 수정하지 않는다. 역할·shell·witness·relay 규칙, 원본 sampled view 내부 선택, induced directed edge/속성/순서/중복 보존도 그대로다. 경로 손실을 0으로 만들기 위한 relay 증가나 자동 seed tuning은 하지 않는다.

모델은 원본 v1.0의 10,434,532 parameters, 3층 local GAT, 원본 CNN/L1/L2/scalar head/loss다. v1 source anchor GT, 8-candidate curriculum, 후보 pool128, 두 view와 전체 source mask를 유지한다. Basic CP80와 v2.2의 관측 P/U를 변경하지 않는다.

## 104와 416의 검증 범위

r3의 **실제 runner smoke는 104 profile만**이었다. 그보다 앞선 `nested416` 결과는 dedicated graph audit/checker의 결과다. 두 검사 경로를 같은 것으로 부르지 않는다.

이번에 추가하는 실제 runner smoke의 명시적 profile은 `64/32/96/64/96/64`다. 역할 순서는 tumor surface/interior, source context/liver surface, target context/liver surface다. 416은 seed 예산 합이며 최종 node/edge 상한이 아니다. 전체 node 수에는 남은 원본 relay가 포함된다.

검토에서 416은 일부 구조 손실을 줄이는 보수적인 연구 출발점으로 제안됐다. 이것은 정확도 우월성 증명이나 필수 production 기본값이 아니다. 104보다 약간 나쁜 방향별 경로 지표도 있으므로 모든 손실이 작아졌다고 주장하지 않는다. 첫 장기 대조에서는 native와 명시적으로 선택한 profile 하나만 비교한다.

추가 GPU smoke는 기존 실제 CT fixture와 원본 model을 사용하고 physical 2 samples × 8 candidates, 두 view, spawned workers2, branch당 1 update다. VRAM12 GiB/RSS32 GiB의 별도 DEBUG이며 전체 84/21 데이터 설정과 40epoch를 덮어쓰지 않는다. 새 결과는 `work/v1_sampling_runtime416_CUDA_DEBUG_20261003`에 기록하고 기존 104/416 측정·r3 ZIP·실패 기록은 보존했다.

## 추가 실제 GPU 결과

양쪽 실제 runner smoke는 PASS다. 동일 초기 neural state는 r3와도 일치했고 측정한 구현 파일 hash는 실행 후에도 모두 그대로다. 핵심 CNN/L0/role·shell pooling/fusion/L1/L2/scalar gradient가 유한한 양수이며 optimizer 변경, validation, chunked scoring, 실제 GPU checkpoint reload와 596 voxel 전체 source-mask paste를 확인했다.

| 이번 동일 입력/초기 state의 검사 | Native | Strict-nested416 |
| --- | ---: | ---: |
| 평균 N / train graph | 13,138.875 | 1,112.3125 |
| 평균 directed typed E / train graph | 889,816.5625 | 18,945.75 |
| N 범위 | 7,161–20,473 | 866–1,440 |
| 1 update compute | 23.8585초 | 2.8188초 |
| Peak allocated VRAM | 6.1762 GiB | 0.5163 GiB |
| 최초 train loader batch | 14.7171초 | 18.1817초 |

평균 그래프 통계는 이번 train batch의 16 graphs/view다. 이전 48-view audit 평균과 분모를 섞지 않는다. 이전104 runtime의 native31.30초와 이번 native23.86초도 별도 실행이므로 profile간 정확한 속도 우열을 만들지 않는다. 간 마스크 coverage 기준은 원본0.85를 그대로 적용했고 선택 위치에서는1.0이었다. Mask 밖 CT/label은 변경되지 않았다. 이 검사는 여전히 DEBUG 8개 중심이며 실제 production128 후보/transform 검증이나 추천 정확도 평가가 아니다.

원시 결과와 자원·scope는 `validation/v1x_progressive_20261003/runtime416_verification.json`에 결속했다. 원본 archive와 sampler는 기존 SHA256 그대로다. 독립 GPT 검토의 CPU 검사 수나 r3의85개 UNIT 검사 수를 이번 새 검사 수로 합산하지 않았다.

추가 자료 결속 검사8개도 통과했다. 104 결과를416로 바꿔 넣기, profile/초기 state/실행 후 sampler 변경, smoke의 품질 통과 승격을 거부한다. 이8개는 파일·JSON 검사이며 실제 CT/CUDA 성공과 별개다. r4는 전달 ZIP의 개정 번호이고 새로운 모델 버전이 아니다. 모델 stage는 v1.0, 연구 방향은 v1.x sampling 대조로 유지한다.

## 난수와 비용의 해석

worker calibration 전후 RNG 복원, 동일 seed42·초기 neural weights는 유지한다. 그러나 graph node/edge 수가 달라 원본 GAT attention dropout tensor 크기가 달라진다. 이후의 dropout mask trajectory까지 같다고 주장하지 않는다. 모델 dropout을 꺼서 대조 조건을 바꾸거나 gradient 경로에 난수 동기화를 새로 넣지 않는다. 작은 single-seed 차이는 stochastic training 변동과 분리되지 않으며, 애매한 품질 결과에서만 추가 seed를 검토한다.

compute update 시간은 synchronized forward/loss/backward/gradient 검사/clip/optimizer 구간이다. 첫 update의 초기 할당이 포함되고 sampler·loader·H2D·전체 validation·checkpoint IO는 포함하지 않는다. 실제 runner는 native materialization 뒤 추가 thinning을 하므로 CPU 준비 비용이 남는다. 계산 시간이 줄어도 전체 epoch가 같은 배율로 줄었다고 말하지 않는다. **v1에는 v2.2식 support refresh가 없다.**

## 장기 비교의 실행 계약과 미검증 범위

native 실험이 shared prototype/cache 준비를 소유한다. nested는 그 native를 reference로 지정하며 native의 실제 calibration 이후 같은 physical batch/worker lock을 사용한다. 새 계약의 source snapshot과 checkpoint는 기존 legacy·104 DEBUG와 별도다. 두 arm의 모델 stage는 모두 v1.0이며 sampling mode/profile 차이를 별도 계약으로 기록한다.

416 initializer는 이미 다음 값을 지원한다. 이것은 **설정 예시이며 학습 시작 명령이 아니다**.

```text
--local-sampling strict_nested
--role-seeds 64 32 96 64 96 64
--reference-experiment <새 explicit-native 실험>
```

양쪽 전체 40epoch 완료 후에 full21 evaluator로 같은 case의 MRR/top1/margin 차이와 case 단위 paired uncertainty를 산출한다. 허용 quality drop은 미정이며 임의 통과 기준을 추가하지 않는다. 전체128 production generation/transform 검증과 Basic CP80 대비 nnU-Net 실험은 그 뒤 별도 범위다. 이번 작업에서는 장기 학습·전체21 평가·전체128 generation·서버 접속·Git push를 수행하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 승인된 명시적 sampling 대조만 사용한다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. OOM을 숨기는 fallback을 추가하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 새 profile smoke의 결과는 별도 원시 보고서에 결속한다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
