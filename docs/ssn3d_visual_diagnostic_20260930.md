# SSN 3D 단일 CT 그래프 시각화 — 독립 DEBUG

요청: SSN으로 실제 그래프를 만들고 voxel 영역과 슈퍼노드 연결을 시각화한다.
기존 same-donor 서버 준비/학습, Basic CP, L1/L2, 원본 loss와 실험 설정은 변경하지 않았다.

## 원문과 구현 범위

- [SSN, ECCV 2018](https://arxiv.org/abs/1807.10174), Algorithm 1과 §4.
- [공식 Caffe 구현](https://github.com/NVlabs/ssn_superpixels/blob/master/create_net.py), `cnn_module`, `compute_assignments`, `exec_iter`를 확인했다. 공식 저장소 라이선스는 CC BY-NC-SA 4.0. 이번 코드는 방정식의 독립 PyTorch 구현이며 공식 바이너리나 가중치를 사용하지 않았다.
- CNN: 64채널 convolution 6개, 두 번의 pooling, 세 해상도 결합, 학습 특징 15채널 출력 구조를 참고했다. 3D 버전 trainable parameter 640,419개.
- 변경: 2D→3D, RGB/XY→정규화 CT/XYZ, 주변 중심 9개→27개, trilinear resizing. 최종 배정 특징 19차원. 좌표는 seed-cell 단위이며 명시적인 공간 prior다.
- 원문의 semantic reconstruction 대신 **CT reconstruction MSE + 0.01 × soft coordinate reconstruction MSE**를 사용한 별도 진단이다. 원문 compactness의 hard reconstruction과도 다르다. 원 논문의 학습 재현이나 CT 의미 분할 모델이라고 부르면 안 된다.
- 최종 hard assignment를 6-connected component로 분리한다. 작은 성분을 삭제하거나 이웃에 강제로 합치지 않는다. 따라서 최종 영역 수는 초기 중심 수보다 많을 수 있다.
- 영역에 속한 19D 특징의 평균을 node attribute로 저장한다. 면을 공유하는 영역에만 무방향 edge를 생성한다. radius/kNN 재연결은 없다.
- Soft assignment/집계까지 미분 가능하다. hard connected-component graph는 시각화 추출 결과다. 새 GNN 및 CP ranking loss까지 통합했다고 주장하지 않는다.

## 데이터·명시적 진단 설정

실제 `liver_66`의 기존 후보 `liver_66:1`, `liver_66:19` 두 입력을 physical batch 2로 동시에 처리했다. 각각 전체 기존 48³ patch이며 native CT 전체가 아니다. 종양 mask는 네트워크/손실/배정에 넣지 않았다. 원본 데이터·mask를 바꾸지 않았다.

- 같은 48³ 입력에서 8-voxel seed cell → 6³=216개 초기 중심. 국소 영역 구조를 살펴보기 위한 명시적 진단 profile이며, 최종 연구 설정이나 node 상한이 아니다. 출력 수를 보고 값을 다시 조절하지 않았다.
- SSN assignment 10회, 새 초기 가중치, Adam lr 1e-4, 32 update의 동일 CT 재구성 검사. 데이터 분할 성능 검증이 아니며 단일 CT fitting이다.
- RTX 5070 Ti 16 GiB, FP32, allocator 한도 6 GiB, 4 CPU workers, 16 logical CPU. 실행 시작 여유 RAM 약 38 GiB. GPU 프로세스·서버 연결을 종료하지 않았다.
- physical/effective batch 모두 2, accumulation 1. 요청한 단일 CT의 두 기존 view를 함께 처리한 진단이며 production batch 선택 실험은 아니다.
- 전체 update CUDA 동기화 시간 합 14.56초, peak CUDA 1.70 GiB, RSS 약 1.83 GiB. 공식/서버와 처리량 비교값이 아니다. report의 forward 구간은 GPU 동기화 없는 host 실행 구간이므로 독립 GPU forward 비용으로 해석하지 않는다.
- 첫 실행은 psutil WindowsPath 형식 오류로 학습 전에 중단됐다. 경로를 문자열로 수정한 r1 결과를 사용했다. 실패 폴더를 삭제하지 않았다.

## 실제 결과

| 후보 | 상태 | 영역/노드 | 무방향 edge | 1-voxel 영역 |
|---|---|---:|---:|---:|
| liver_66:1 | 초기 | 251 | 1,118 | 30 |
| liver_66:1 | 32 update 후 | 257 | 1,148 | 36 |
| liver_66:19 | 초기 | 451 | 1,627 | 136 |
| liver_66:19 | 32 update 후 | 433 | 1,594 | 119 |

각 결과는 110,592/110,592 voxel을 포함하며 RAG 연결성분은 1개다. 배정 합 오차 최대 2.38e-7, 비국소 배정 질량 0. 전체 trainable parameter의 첫 gradient가 finite/nonzero이고 CNN 가중치가 실제 갱신됐다.

동일 입력의 eval CT reconstruction MSE는 0.045815→0.043997. 이는 종양 Dice, 조직 분할 품질, 추천 성능, 일반화 성능이 아니다. 시각적으로 grid initialization 영향이 남고 작은 성분이 다수 존재한다. 이를 품질 PASS로 표시하지 않는다. 기존 fine graph는 공간 역할별 위치 표본이고 이번 그래프는 full patch 영역이므로 노드 수만으로 동등한 문맥 보존 또는 가속을 주장하지 않는다.

## 파일과 검사

- 구현: `tools/ssn3d_diagnostic.py`
- GPU 실행: `tools/run_ssn3d_visual_debug.py`
- 단위 검사: `tests/test_ssn3d_diagnostic.py`의 5개 함수를 직접 실행. 환경에 pytest가 없어 테스트 함수를 Python으로 호출했으며 모두 통과.
- 검사항목: 배정 확률/범위/gradient, 명시적 거리식 대조, 연결성분 분리·voxel 보존, face adjacency/특징 평균, autograd gradcheck.
- 실제 결과: `work/ssn3d_visual_DEBUG_20260930_r1/report.json`, 원본 label/center/feature/edge/size의 NPZ 4개. 모든 graph 저장은 DEBUG 폴더에 한정된다.
- 시각화: `tools/ssn3d-regions.template.html`, `tools/export_ssn3d_visual.py`.
- 브라우저: `tools/check_ssn3d_visual.cjs`, `visual_checks.json`, `ssn3d.png`, `ssn3d-second.png`, `ssn3d-mobile.png`. 후보/초기-학습 후/단면/회전/영역 선택 및 328px 폭 확인.
- production checkpoint, ready 표시, 전체 학습·평가, Git push 없음.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 요청한 단일 사례 DEBUG이며 두 patch의 모든 voxel을 사용했다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 진단 범위와 초기 seed 설정을 명시했다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 두 view를 GPU batch로 처리했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실행 OOM 없음.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다. 실제 CT를 사용하고 초기화/학습 범위를 명시했다.
- [x] 구현한 CNN·soft assignment·집계가 forward, reconstruction loss, gradient와 optimizer에 연결되어 있다. CP/GNN 통합은 범위 밖이다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
