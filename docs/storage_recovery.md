# 기존 캐시 공간 회수

2026-10-08. 사용자 서버에서 Git objects와 학습 출력 파일 생성 모두 `errno=122`로 거부됐다. 이전에 승인된 구형 그래프 609개는 이미 없으며, 정리 도구 재실행의 추가 회수량은 0이다. 이를 다시 삭제하라고 안내하지 않는다.

## 실제 변경

`tools/reclaim_fields.py`는 Python 표준 라이브러리만 쓰는 서버 정리 도구다. 옵션 없이 실행하면 메타데이터와 inode만 조사한다. `--interactive`는 중복 후보의 실제 SHA256을 확인하고 정확한 계획을 `/tmp`에 저장한 뒤 `SHARE` 입력을 받아 적용한다. 승인 전에는 캐시를 변경하지 않는다.

대상은 `Medical/experiments` 아래 다음 다섯 폴더의 `data/whole_case_fields/<case>/depth.npy`, `occupied.npy`뿐이다.

- `v18_u_bridge_m10_seed42`
- `v18_selected_m10_seed42_memory`
- `v18_native_m10_seed42_memory`
- `v19_native_fixed_m10_seed42`
- `v19_native_listwise_m10_seed42`

같은 case·binding·field·전체 파일 SHA·크기·파일시스템·소유권·권한인 파일만 합친다. 이미 hardlink인 경로는 회수량에 중복 합산하지 않는다. 교체 대상은 link 수가 1인 파일로 한정한다. 소유자가 현재 사용자와 다른 파일은 적용하지 않는다. signed metadata와 원본 파일 내용이 변했으면 중단한다.

동일 내용을 가리키는 임시 hardlink를 먼저 만든 다음 기존 정확한 경로에 atomic replace한다. **먼저 삭제하거나 큰 파일을 복사하는 우회는 없다.** 기존 파일명·배열 바이트·metadata JSON·체크포인트·모델·원본 CT·그래프는 보존한다. 중간 실패 시 이미 합친 파일은 정상이며, 다시 조사하면 남은 독립 사본만 대상이 된다. Git/학습 쓰기 검사도 정리 후 도구가 실행하며, 자신이 만든 작은 검사 파일만 제거한다.

이 작업은 기존 캐시 inode를 바꾸므로 저장소 규칙 §1의 기존 파일 교체 승인 대상이다. 서버에서 실제 경로와 회수 추정량을 먼저 표시하고 마지막에 명시적으로 승인받는다. 네 학습과 해당 공유 원본 작업이 모든 서버에서 멈춘 상태여야 한다. 잠금 소유자가 살아 있거나 확인할 수 없으면 적용하지 않는다. 다른 프로세스나 세션을 종료하는 기능은 없다.

현재 quota로 홈에 Git 파일을 받지 못하므로, 배포 시 `/tmp`의 별도 Git 저장소로 검증한 revision을 받고 도구 하나만 추출한다. 기존 홈의 Git checkout·실험 경로는 바꾸지 않는다. 정리 명령에 학습 시작을 연결하지 않는다.

## 재발 방지

`preparation_reuse.py`의 기존 잠금은 arm별 목적지에만 걸려 있었다. 서로 다른 두 실험이 같은 환자의 캐시를 동시에 처음 요청하면 둘 다 원본 배열을 만들 수 있었다.

새 구현은 기본 다섯 캐시 루트가 서로를 조회할 수 있는 설정에서 case·binding별 공유 잠금을 사용한다. 먼저 끝난 한 실험의 완성본을 대기한 실험이 다시 확인하여 hardlink로 가져온다. 다른 환자는 계속 병렬 처리한다. 잠금은 공통 부모의 `.field_publication` 아래 작은 파일로 남는다. 기존 완성된 캐시 내용과 학습 설정은 바꾸지 않는다. 구버전으로 계속 실행하는 프로세스는 이 새 잠금에 참여하지 않는다.

## 검증과 한계

정리 회귀 검사는 실제 임시 파일의 SHA·hardlink·부분 적용 후 재실행·quota 오류 주입을 사용한다. 학습이나 실제 CT를 이용한 성능 실험이 아니다. 공유 생성 검사는 동시 스레드에서 배열 한 벌만 생성되는지, 다른 case가 병렬 진입하는지 확인한다. Linux 다중 프로세스 검사는 별도로 존재하지만 Windows 환경에서는 실행하지 못했다. 실제 NFS의 동시 프로세스 동작과 서버 회수량은 서버 실행 전 미검증이다.

`st_blocks × 512`는 회수 추정치다. 열려 있는 mmap/파일, NFS 처리 및 별도 quota 때문에 실제 사용 가능량과 다를 수 있다. 4KiB 쓰기 성공 역시 미래 checkpoint나 추가 캐시 공간을 예약하지 않는다. 임시 hardlink 생성 자체가 EDQUOT로 거부되면 기존 파일을 보존하고 실패를 보고한다. 이 경우 다른 파일을 임의로 삭제하지 않는다.

## 서버 적용 결과 — 2026-10-08

사용자가 서버 실행 결과를 제공했다. `/tmp/hiercp-fields-77gwxvul.json` 계획에 따라 174개 중복 경로를 공유했고, 대상 inode의 할당 블록 기준 회수 추정량은 85.93GiB였다. 기록은 `/tmp/hiercp-fields-77gwxvul.json.journal`이다. 전체 계획과 journal 원문은 로컬로 가져오지 않았으며, 콘솔 결과를 근거로 기록한다.

- `v19_native_fixed_m10_seed42`: 40개, 20.20GiB 추정.
- `v19_native_listwise_m10_seed42`: 134개, 65.73GiB 추정.
- Git objects, selected, native, native_fixed, native_listwise의 실제 저장 위치에서 4KiB 쓰기 모두 PASS.

이전의 즉시 파일 생성 실패는 이 검사에서 해소됐다. 85.93GiB는 실제 quota 잔여량 측정값이 아니다. 서버 학습 재개, 다음 실제 checkpoint 저장, NFS 공유 생성의 동시 실행은 이 결과에 포함되지 않는다. 정리 도구 실행 revision은 `2441c63316bb0194f88d0a84e09fbe50c4f0224d`다. 같은 revision의 중복 생성 방지 변경은 학습을 이 코드로 재개해야 적용된다. 위 UNIT 검증 당시 상태를 보존한 `validation/storage_recovery.json`과 서버 콘솔 기록 `validation/storage_recovery_server.json`을 구분한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 승인 없이 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] 독립 case 병렬 처리와 같은 case 중복 생성 방지를 검사했다. physical batch는 변경하지 않았다.
- [ ] 서버 GPU, CPU, RAM 활용 상태를 새로 측정했다. 이번 작업은 저장 공간 관리다.
- [x] 모델 축소보다 저장 캐시와 실제 EDQUOT 원인을 조사했다.
- [x] UNIT 파일 검사와 production 학습을 구분했다.
- [x] dummy, placeholder, random fallback을 학습 경로에 추가하지 않았다.
- [ ] forward, loss, gradient와 optimizer를 새로 실행했다. 모델 경로 변경 없음.
- [x] 실제 변경 사항과 서버 적용 전 미검증 범위를 명확하게 보고했다.
- [x] 저장 관리 검사와 전체 학습·전체 평가를 구분해서 보고했다.
