# v1.7 전환 실험 계약 수정: A/B에서 빠진 변경점 포함

기존 A/B는 전체 v1→v2.2 변경점을 이분한 실험이 아니었다. A는 원래 v1 입력에 CNN을 연결했고, B는 원래 v1 과제에 v2.2 상위 연산을 연결했다. 입력, 정답, 전체 후보 학습, loss, support 및 학습 정책의 상당 부분은 그대로였다. 따라서 A/B 결과는 보존하지만, 두 실험이 학습됐다는 이유로 모든 변경 부품을 검증했다거나 후보 개수만 남았다고 결론내리지 않는다.

이 문서의 v1.7은 전환 실험 계약의 버전이다. 새로운 학습 모델이나 전체 전환 arm의 실행 완료를 뜻하지 않는다.

## 전체 변경 범위

`hiercp_v1x/transition_inventory.py`가 전체 변경축과 A/B의 실제 적용 범위를 관리한다. 일부만 옮긴 항목은 partial로 남긴다. 전체 이분 설계는 다음 두 묶음에서 항목이 빠지거나 중복되지 않아야 한다.

현재 목록은 29개 축이다. 기존 A/B가 완전히 대조한 축은 5개, 일부만 대조한 축은 6개이며 18개는 미대조다. 따라서 기존 A/B에 포함되지 않은 차이를 128후보 평가 하나로 설명할 수 없다. 새 설계 JSON은 입력·표현·forward 11개 축과 과제·학습·평가 18개 축을 중복 없이 모두 포함한다.

- **입력·표현·forward:** physical ROI와 해상도, 영상 정규화, target erasure, CNN/로컬 그래프, role·shell readout, pair fusion, L1 관계 및 label topology, L2 prototype 구성과 scalar score.
- **과제·학습·평가:** source/donor 배정, 실제 샘플 inventory와 GT 정의, 8후보 대 전체 P/U 후보, loss 항목과 정규화, support 정답·제외·refresh, 두 view와 P×U tile, batch 단위, precision·LR 정책, validation 후보 및 BEST 선택 지표.

GPU 종류, 메모리 한도, worker 및 저장 경로는 실제 실행 기록에 결속한다. 자원이나 batch 숫자가 같다는 이유로 서로 다른 sample 단위를 동등한 작업량으로 간주하지 않는다. 성능을 맞추려고 margin, 후보, 모델 깊이, channel, epoch, 데이터 범위를 자동 변경하지 않는다.

## GT와 평가의 경계

v1은 source의 원래 anchor와 선별·관계 corruption 후보를 비교한다. native v2.2는 관측 적격 종양 P와 미관측 U를 비교하며, U는 CP 부적합 정답이 아니고 P는 외부 donor 적합도 정답이 아니다. 두 GT를 서로 바꾸거나 같은 정확도의 분모라고 부르지 않는다.

원본 v1의 validation은 전체 난이도를 활성화한 고정 8후보 평가다. native v2.2의 128은 U 개수이며 P까지 포함한 inventory를 학습과 평가에 사용한다. 원본의 sample별 정답 rank와 native의 환자별 첫 P rank/관측별 recall은 서로 다른 지표다. 기존 v1 MRR=1을 native P/U 평가의 합격 컷으로 재사용하지 않는다.

전체 변경점을 포함한 교차 arm은 원래와 다른 입력 인터페이스 및 GT를 실제로 연결해야 한다. 원본 L0가 요구하는 역할별 표현을 단일 CNN 벡터로 복제하거나, 이름이 `V1LocalEncoder`인 CT-only adapter를 원본 v1 L0로 부르면 안 된다. 원본 graph를 새 donor/recipient 자료로 생성하는 경로와 native crop을 원래 curriculum 자료로 읽는 경로가 필요하다. 아직 이 연결이 실행 검증되지 않은 arm은 runnable로 등록하지 않는다.

공통 held-out 평가를 구성할 때는 후보·입력·GT·분모를 별도 계약으로 고정한다. 원래 과제의 유지 평가도 함께 기록하되 다른 과제의 MRR 절대값 차이만으로 원인을 판정하지 않는다. 두 절반이 개별적으로 실패를 재현하지 않을 때 단일 고장 부품을 만들어내지 않는다.

## 지금 실행 가능한 경로

`tools/export_v1_v22_transition_receipt.py`는 이미 알려진 native 실패 실험과 baseline/A/B의 실제 설정을 읽는 자료 수집 명령이다. GPU forward, backward, optimizer, 학습 재개 또는 CP를 실행하지 않는다. 기존 CT, checkpoint, 결과 파일은 읽기만 하고 새로운 출력 폴더에 자료를 보존한다.

native의 experiment manifest에 결속된 checkpoint를 선택한다. 임의 최신 파일 검색은 하지 않는다. checkpoint의 전체 tensor 값 대신 metadata를 읽고, 실제 inventory와 실행 설정, epoch 기록, source SHA를 서버의 새 폴더에 보존한다. 기본 CLI는 ZIP을 만들지 않는다. `--archive`를 명시한 경우에만 서버 내부에 별도 압축본을 만든다. CT·주석·가중치 파일은 복사하지 않는다. 현재 코드가 saved source identity와 다르거나 필수 실행 정보가 없으면 미결 항목으로 보고하고 exact reproduction/production ready로 승격하지 않는다.

대상은 `/home/aicompetition06/Medical/experiments/v22_cnn_m10_seed42`이다. baseline은 `v1_m10_seed42_20261004`, A는 `v15_m10_halfA_seed42_20261004`, B는 `v16_m10_halfB_seed42_20261004_r3`이다. 사용자에게 같은 경로를 다시 묻지 않는다.

`bash tools/run_v17_transition_terminal_server.sh` 한 번으로 알려진 네 실험의 자료 수집과 터미널 요약 출력을 수행한다. 사용자에게 ZIP 다운로드나 첨부를 요구하지 않는다. 요약은 실제 batch·epoch·step, input/GT/loss/support, 전체 inventory 개수, 학습 기록, 미결 출처 항목을 보여준다. 긴 목록은 출력만 집계하며 전체 원문과 JSON은 서버에 보존한다. console용 JSON은 원래 receipt manifest와 분리해 옆 파일로 기록한다. 출력 줄 수 제한은 모델·데이터 규모 제한이 아니다.

실제 서버 receipt를 받기 전에는 현재 개발 checkout을 실패 실험의 정확한 학습 recipe라고 가정하지 않는다. 이번 변경은 전체 누락 목록과 실제 recipe를 확인할 실행 경로이며, 전체 전환 학습 arm의 완료 보고가 아니다. 기존 A/B 전체 학습, 로컬 smoke, 자료 결속 검사를 서로 구분한다.

## 이번 변경의 검증

2026-10-05에 `tests.test_v1_transition_inventory` 23개와 `tests.test_native_transition_receipt` 23개, 총 46개의 metadata/contract 단위 검사를 실행해 통과했다. 실제 PyTorch checkpoint serialization의 metadata 로딩, 전체 14,102개 fixture inventory 보존, 모든 epoch 기록 수집, 원본 파일 보존, source 불일치와 누락의 명시적 보고를 포함한다. wrapper 설정·실제 batch·저장된 epoch/step/cursor·학습 schedule을 대조하고 수집 종료 시 checkpoint SHA를 다시 검사한다. fixture는 단위 검사 입력이며 실제 CT나 서버 성능 결과로 사용하지 않았다.

CLI help도 실행했다. 새로운 전체 전환 arm의 CT forward/backward, optimizer update, GPU smoke, 서버 학습, 공통 평가를 실행한 것은 아니다. 단위 검사 통과를 이들 단계의 완료로 표시하지 않는다. 실제 실패 실험의 recipe 결속과 양쪽 입력 bridge 연결이 남아 있다.

같은 날 터미널 출력 경로를 추가한 뒤 inventory 23개, receipt 24개, terminal reporter 16개, CLI 3개, 총 66개 metadata 단위 검사가 통과했다. 표시 경로를 실제 sibling JSON 경로로 맞춘 최종 수정 후 reporter·CLI 19개도 다시 통과했다. 서버 wrapper의 shell syntax 검사와 CLI help를 확인했다. 기본 실행은 ZIP을 만들지 않고 알려진 네 실험의 기존 설정·결과를 터미널에 출력하며, 전체 자료는 서버에 보존한다. 실제 서버에서 이 새 명령을 실행하거나 새로운 neural 비교를 수행한 결과는 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. sample 단위 차이를 실험 요인으로 기록한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 CUDA 증거와 서버 수집 metadata를 구분한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 경로는 학습 실행이 아니다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 새 전체 전환 arm의 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 아직 구현·GPU 검증 전이며 runnable로 표시하지 않는다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
