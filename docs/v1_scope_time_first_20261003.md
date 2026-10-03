# v1 공간 범위 축소: 시간을 먼저 확인하는 첫 실험

## 수정한 실험 순서

사용자의 첫 실험은 **실제로 보는 공간 범위를 줄여 처리시간을 줄이고, 그 결과가 유효한 뒤 다음 교체 실험으로 넘어가는 것**이다. 공간을 그대로 둔 native sampled graph에서 노드만 줄이는 strict-nested416은 이 첫 실험과 다르다. r5 ZIP과 그 검사는 보존한다. 416의 GPU 계산 감소를 공간 범위 축소 완료나 전체 epoch 가속으로 해석하지 않는다.

직전에 제시한 `run_v1_native_nested416_server.sh`의 native40 → nested40 자동 진행 명령은 현재 첫 실험에 사용하지 않는다. 새로운 범위 진단은 장기 학습을 호출하지 않는 별도 entrypoint다.

## 실제 범위를 제한하는 계약

실행에서 `--margin-mm 10 20 30`처럼 mm를 필수로 지정한다. 임의 기본값이나 OOM 후 자동 조정은 없다.

- ROI: 전체 원본/변환 종양 footprint bbox에 각 축 양쪽으로 지정한 margin을 더한다. Native spacing에서 ceil로 올림하고 홀수 정렬하며, 필요한 경우 한쪽에 추가 1 voxel이 생긴다. 양쪽 실제 padding을 mm로 각각 기록한다. 종양 mask voxel을 자르지 않는다.
- Context node: 실제 간 실질 내부에서 종양으로부터의 거리 2mm 이상, 지정한 margin 이하만 사용한다.
- 원본의 `max(30, context_outer+2, depth+4)` 확대와 간 표면을 찾기 위한 추가 ROI 확장을 적용하지 않는다. 깊은 후보도 지정한 범위를 유지한다.
- 범위 안에 실제 간 표면이 없으면 두 liver-surface 역할에 실제 node가 없다는 사실을 기록한다. 다른 조직을 간 표면으로 바꾸거나 바깥 표면을 가져오거나 sample을 건너뛰지 않는다. 존재하는 표면을 삭제하지 않는다.
- 원본의 좌표 및 SDF 정규화 28mm, shell 경계 4/12/28mm, 물리 grid spacing, relation 반경, GAT 및 모델 깊이/너비/parameter 수는 유지한다. ROI와 annulus 외곽만 실험 변수다. 좁은 범위에서 빠지는 shell은 원본에 이미 있는 empty-shell 표현을 사용한다.
- CNN 입력은 5×48³, full mask, candidate pool128, 실제 학습 candidate8, 두 view, 원본 positive/corruption GT와 원본 loss를 유지한다.

실제 표면이 없는 graph의 role pooling은 전체 graph 수를 명시한다. 이는 없는 표면을 가짜 node로 표현하는 대신 구조적 부재를 처리하기 위한 bounded profile의 명시적 확장이다. 원본과 수학적으로 동일한 local graph, 기존 native checkpoint의 exact resume라고 주장하지 않는다. 별도의 scope identity가 canonical payload, sampled graph, 모델 architecture에 결속된다.

원본 context outer는 28mm다. 30mm arm은 명시한 30mm annulus를 사용하므로 원본 sampled graph의 strict subset이라고 표현하지 않는다. 첫 GPU 실행 r2에서는 SDF 분모도 margin에 따라 바뀌는 혼입을 발견했다. r2 결과는 보존하지만 최종 범위 대조의 근거는 SDF 분모를 28mm로 고정한 r3 결과다.

CNN CT 채널의 간 밖 값 처리 방식은 보존 원본 v1과 같다. Graph context node의 간 내부 조건과 CNN 영상 채널의 masking은 서로 다르며, 이 진단이 간 밖 CNN 신호까지 배제했다는 주장은 하지 않는다.

## 첫 실행의 범위

`tools/run_v1_scope_probe.py`는 기존 native cache의 성공한 기록 중 명시한 physical batch와 별도 validation case의 원본 바이트/CT provenance를 확인한다. 불완전한 native 준비를 완성으로 승격하지 않는다. 기존 후보 중심·변환·정답·patient/population 입력을 고정한 채 원본 CT에서 각 margin의 ROI를 다시 만든다. 큰 ROI 캐시를 잘라서 작은 ROI의 실제 준비 비용처럼 보고하지 않는다.

이 실행은 명시적인 DEBUG다. 최종 84/21 데이터셋은 줄이지 않으며, 선택한 일부 case의 짧은 비용·forward·gradient 검사를 전체 품질 평가로 표시하지 않는다. 첫 실행은 한 개의 동일한 physical mini-batch와 전체 8 candidate × 두 view에 대해 각 margin의 실제 준비, sampling/collate, H2D, forward/backward/optimizer, validation 시간을 구분한다. GPU 및 RSS 한도, CPU worker 수는 실행에서 지정한다. 모든 실패는 보존된 report와 함께 드러내며, margin/batch/모델을 자동 변경하지 않는다.

Production checkpoint, ready marker, 40-epoch 학습 및 nnU-Net 학습은 만들거나 시작하지 않는다. 준비 일회 비용과 update 비용을 분리하며 작은 DEBUG 시간에서 84/21 epoch 비용을 단정하지 않는다. 비용 결과가 나온 뒤에 공간 범위 후보의 학습 유효성을 검증한다.

한 batch에 대한 첫 update이며 warmup update는 없다. 따라서 kernel 초기화 및 gradient 검사 비용이 포함되며, production loader queue·checkpoint 저장·전체 epoch를 측정한 결과가 아니다. 로컬에서 다른 GOIS 학습 PID 10524가 동일 GPU process 목록에 나타났다. 다른 작업을 중단하지 않았으며, 로컬 시간은 GPU 공유 상태의 참고 측정으로 보고한다. 독점 서버에서 동일 범위의 짧은 비용 비교를 별도로 실행할 수 있다.

## 검증 상태

구현과 검사 결과는 별도의 실제 실행 receipt에 기록한다. 이 문서만으로 GPU 검사 통과, 시간 단축 또는 추천 품질 통과를 선언하지 않는다. 기존 strict-nested416 검사는 별도 증거로 그대로 보존한다. 보류한 deferred-edge 실행 최적화도 첫 공간 범위 비교에 섞지 않는다.

최종 r3은 실제 liver_5/liver_6의 동일 physical sample batch2(후보 graph16 × 두 view)와 별도 liver_31을 사용했다. 모델은 원본 10,434,532 parameters, 모든 arm의 동일 initial state hash, 후보8/pool128, 원본 GT·L1/L2·loss·전체 source mask를 유지했다. CNN·GAT·role/shell pooling·fusion·L1·L2·scalar head 모두 유한한 양의 gradient norm을 기록했고 모든 trainable parameter1085개에 gradient가 있었으며 optimizer가 score head를 변경했다. 전체 학습·전체 평가·추천 품질 검증은 수행하지 않았다.

| Margin | 평균 ROI voxel (source/target 27개) | 후보 graph 평균 N/E (train16개, 한 view) | ROI+canonical 준비 | 두 view+collate+pin | 첫 update (검사 포함) | peak PyTorch VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10mm | 103,766 | 4,628 / 373,628 | 9.72s | 4.40s | 14.18s | 2.45GiB |
| 20mm | 310,644 | 10,832 / 795,182 | 13.78s | 7.52s | 23.01s | 5.27GiB |
| 30mm | 704,729 | 13,038 / 896,937 | 21.83s | 9.23s | 29.58s | 6.16GiB |

Raw CT/hash/depth 비용은 세 case 합계65.6/65.2/68.1s로 별도다. 이는 좁은 ROI와 무관한 전체 원본 CT 읽기·depth 준비이며 매 update 비용이 아니다. RSS peak는8.91/11.09/11.66GiB, GPU는 RTX5070Ti16GiB, 명시적 CUDA budget12GiB·RSS budget32GiB·workers2다. GPU 공유와 첫 kernel 실행이 포함된 위 시간으로 배속 또는 epoch 시간을 확정하지 않는다. 단위 검사는37개 PASS이며 별도 synthetic geometry integration의20개 contract assertion을 포함한다.

원시 r3 보고서: `validation/v1x_progressive_20261003/scope_time_debug_report.json`. 결속 및 미검증 상태: `validation/v1x_progressive_20261003/scope_time_debug_release.json`. 초기 loader 실패 r1과 SDF 혼입을 찾은 r2 출력도 기존 work 폴더에 보존했다. 실행이 가능한 bounded scope 구현·정적/단위/실제 CT CUDA smoke 완료와 전체 ranking 품질 검증은 구분한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 사용자가 명시한 공간 범위 축소만 별도 profile로 구현한다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 소수 입력은 명시적 DEBUG이며 최종 설정이 아니다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 동일 batch와 병렬 ROI 준비를 사용한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실제 실행 receipt가 수치의 근거다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 실제 bounded profile r3 CUDA 검사로 확인했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
