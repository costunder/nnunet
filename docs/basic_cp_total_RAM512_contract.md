# Basic CP 전체 RAM 512GiB 실행 계약

사용자가 승인한 운영 변경은 전체 실험 RAM 상한 512GiB와 실제 비어 있는 GPU에서만 새 연산을 시작하는 것이다. 이 문서는 실행 코드의 계약이며 서버에 적용됐다는 완료 보고는 아니다.

## GPU 시작 조건

각 독립 arm은 지정된 물리 GPU UUID를 사용한다. 실행 직전 CUDA compute PID와 소유자를 다시 확인한다. 다른 작업이 있으면 대기한다. 다른 사용자의 작업이나 세션을 종료하지 않는다. 기존 자신의 CUDA 작업에도 중복 학습을 시작하지 않는다. CPU 후보 생성의 최초 시작도 지정 GPU의 빈 상태를 확인한다.

## 전체 메모리 계정

현재 7개 controller와 자식의 RSS, 같은 계정의 보조 프로세스 RSS, `/dev/shm/aicompetition06_basic_cp_ablation_20261013` 아래 파일의 실제 할당 블록을 함께 계산한다. hardlink는 inode당 한 번 계산하고 삭제됐어도 열려 있는 파일은 포함한다. 읽지 못한 자원 정보를 0으로 대체하지 않는다. 파일 할당은 swap을 포함할 수 있으므로 보수적인 상한 계정이다.

| 단계 | 파일 할당 | CPU 후보 생성 RSS | native 작업 RSS | 보조 RSS | 총 예약 상한 |
|---|---:|---:|---:|---:|---:|
| 후보 생성 대기·실행 | 144GiB | 320GiB | 합계 32GiB | 16GiB | 512GiB |
| 전체 후보 완료·자식 해제 후 | 272GiB | 0GiB | 작업당 32GiB, 최대 7개 | 16GiB | 512GiB |

시작 전 모든 arm이 같은 ledger와 SHA가 고정된 source pending 영수증을 사용한다. 실제 642개 source에 대한 512/1024/2048 후보 완료와 producer·worker·tracker의 해제를 확인한 뒤에만 두 번째 단계로 전환한다. 자원 대기는 mutex 밖에서 처리하고, 실행 전 자원 예약과 child의 정확한 PID·생성 시각·UID·명령을 연결한다. 완료 archive는 기존 BEST와 FINAL을 독립 복사하며 복사 중 증가하는 파일 할당도 계정에 포함한다.

## 보존하는 연구 설정

7개 arm: CP 5/7/10개, 후보 512/1024/2048개, donor 50% constant blending. 공통 642개 병변·81명 CP 가능 환자, 전체 105명 학습·26명 평가, 128³ 입력, physical/effective batch 2, accumulation 1, 250epoch, DA worker 4, 후보 생성 worker 24를 유지한다. 후보 확장은 기존 128개와 모든 점수·배열의 prefix를 보존한다. 불가능한 첫 위치는 승인된 동일 환자·동일 병변의 기존 128개에서 재추첨한다. 학습·평가 모델, optimizer, GradScaler와 8개 native logger 배열을 축소하거나 대체하지 않는다.

기존 과학 계약의 native RSS 96GiB·공유 후보 RSS 384GiB 선언은 출처 기록으로 유지한다. 새 운영 필드에 실제 32GiB·320GiB 상한과 전체 512GiB 계정을 명시한다. 새 코드 export를 만들고 기존 실행 코드 파일을 덮어쓰지 않는다. 중단이 필요한 자신의 작업은 정확한 PID·생성 시각·UID·명령을 확인하고 완료된 epoch checkpoint에서 재개한다.

## 완료 및 보존

250epoch FINAL은 학습 완료 증거다. 평가는 자기 BEST로 전체 26명에 수행한다. 이미 완료한 BEST26을 이어받을 때는 원래 fold·checkpoint inode와 SHA가 고정된 원래 validator code export를 사용하고 기존 평가 영수증을 수정하지 않는다. 결과 schema 1과 UI 2, native nnU-Net 로그 디자인을 유지한다.

rolling durable cache는 실제 전체 BEST 모델·optimizer·scaler·logger를 복사한다. 훈련 중 BEST의 동일 inode가 쓰이는 순간에는 변화한 파일 메타데이터를 확인한 경우에만 복사를 보류하고 이전 유효 CURRENT를 유지한다. 변하지 않는 손상이나 pinned 평가 파일의 변경은 오류로 보고한다. 작은 CPU DEBUG fixture 결과를 실제 학습 완료로 보고하지 않는다.

## 작업 완료 체크리스트

- [ ] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [ ] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [ ] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [ ] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [ ] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 실제로 검토했다.
- [ ] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다.
- [ ] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다.
- [ ] 디버그 설정과 최종 설정을 분리했다.
- [ ] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다.
- [ ] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [ ] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
