# v1.7 D: 실제 얇은 간 부위의 빈 context 처리

## 실제 실패와 재현

서버 `aa280829d018ed0f4426a03d59f794b724d76bb0` 준비는 완료 관측835개를 보존한 뒤 `liver_106:2`에서 중단됐다. 이 행은 기존 U (`target=0`)이며 donor는 `liver_117` component14, recipient center는 `[164,302,444]`이다. `liver_106`은 원본 inventory에 관측 종양이 없는 환자다. GT, 중심 또는 donor를 바꾸는 문제가 아니다.

같은 실제 CT/원본 주석/spacing으로 오류를 재현했다. Recipient ROI는45×49×39 native voxel이며 간 voxel3599개, regridded full donor footprint538개, 2–10mm 간 annulus1567개다. 이 annulus의 최대 실제 간 표면 깊이가3.2mm여서 원래 조건인 `organ & ~footprint & distance>=2 & distance<=10 & liver_depth>4`를 만족하는 voxel은0개다. 실제 간 표면 voxel3143개와 canonical surface node78개는 존재한다. Donor source context는445개로 정상이다.

기존 canonical discretisation은 모든 occupied physical cell에서 실제 voxel을 하나씩 보존한다. 따라서 이0개는 무작위 sampling, lattice phase, node cap 또는 누락 때문이 아니다. 실제 깊은 문맥이 없는 경우를 mandatory-role 오류로 처리한 것이 실행 실패의 원인이다.

## 수정 범위

별도 D adapter는 **recipient의 전체 semantic context mask가 실제로 비고 실제 간 표면이 존재할 때만** 빈 target context를 허용한다. 좌표 검사는 target preparation call의 ContextVar 안에 제한하며 worker 간 권한이 섞이지 않는다. Source context는 필수다. 존재하는 context를 빼거나, surface까지 없는 입력을 통과시키지 않는다. 원본 coordinate/finite/간 내부/semantic-role/edge endpoint 검사도 유지한다.

빈 recipient는 실제 surface CT 특징과 원본 모델에 이미 있던 학습 가능한 `empty_context_shell` token을 사용한다. 전체 batch의 target context가0개일 때 token 경로 앞에서 예외를 내던 guard만 target에 한정해 수정했다. 새 dummy node, 임의 상수 예측, node drop, 관측 skip, 범위 확장 또는 fallback은 추가하지 않았다. Mask 부재 증거는 canonical record와 audit/content hash에 묶이고 두 sampled view의 PyG metadata로 전달된다.

`bounded_scope.py`, `local_contract()`, archive와 neural parameter 구성은 변경하지 않았다. 10mm,48³ five-channel dense input, GAT3×128D/4heads, 두 genuine view, 기존 L1/L2와 native loss, 전체14102관측/128U,40epochs, production physical32와 worker16은 유지한다. Nonempty canonical payload/audit는 기존 byte 경로 그대로다. 새 그래프 metadata 검사에는 추가 GPU 동기화가 있으므로 실행 비용이 완전히 같다는 주장은 하지 않는다.

## 완료835개 보존

기존 fastprep의 `canonical_cache/completed` 및 압축 graph/shared-source 파일은 읽기 전용으로 보존한다. Importer는 정확한 공개 `aa28082` source inventory176개와 old local module SHA를 확인하고, 새 recipient adapter로의 명시적인 호환 이전만 허용한다. 완료 행의 source/target context가 모두 nonempty인지 실제 payload를 한 번씩 병렬 읽어 검사한다. GT/center/donor/ordinal/scope/content/file SHA가 일치해야 하며 파일은 그대로 hardlink한다. 원본 CT나 canonical graph를 다시 만들지 않는다.

기존 결과와 새 결과는 별도 experiment root를 쓴다. 전체 coverage를 완성하기 전 index 완료나 checkpoint를 만들지 않는다. 기존 training checkpoint를 이 새 실행의 exact resume로 변환하지 않는다. 현재 서버 실패는 training 이전이므로 이전할 production update는 없다.

## 실제 CUDA DEBUG 결과

RTX5070Ti15.92GiB, PyTorch2.8.0+cu128, CPU16 logical core/torch threads2에서 검증했다. 실제 완료된401개 DEBUG support 관측의 표현을 현재 모델로 새로 계산했으며 실제 `liver_106:2`를 포함한32개 서로 다른 관측/64 genuine graph view를 한 disjoint batch로 처리했다. Mixed32의 native D loss·backward·optimizer는4.554초, peak allocated VRAM6.687GiB였다.677개 trainable parameter tensor 모두 gradient가 있었고 L0/L1/L2/L2_updates의 gradient norm이 모두0보다 컸다. Optimizer 후 neural weight SHA가 실제 바뀐 것도 확인했다.

전체 target context가 빈 실제 관측1개/두 genuine view의 별도 경계 검사와 nonempty 실제 P/U 관측2개의 ranking control도 통과했다. `liver_106`에는 P가 없으므로 두 absence 검사에서 ranking term은0이며, CE/alignment/consistency를 검증한 것이다. 별도 P/U control의 ranking 비교는1쌍이다. 이를 빈 target의 ranking 품질 개선 증거라고 해석하지 않는다.

이 검사는 실제 데이터의 실행 경로 검증이다. 전체14102 server preparation, 서버 A6000 자원 admission,40epoch 학습,21환자 전체 평가 및 CP 추천 품질은 이 로컬 smoke로 완료됐다고 표시하지 않는다. 장기 학습과 nnU-Net은 로컬에서 시작하지 않았다.

실제 report/UNIT 로그/source SHA는 `validation/v17_recipient_context_20261005/verification.json`으로 결속한다. 기존 validation evidence는 변경하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 실패는 OOM이 아니었다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
