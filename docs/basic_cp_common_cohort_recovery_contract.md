# Basic CP 공통 등록부 복구 계약

`tools/recover_basic_cp_common_cohort.py`는 GPU0–6 실험을 기존 800GiB 계약으로
복구하는 운영 도구다. 기존 등록부가 새 프로세스의 RSS를 빠뜨리지 않도록,
새 컨트롤러 7개가 모두 같은 새 등록부를 참조한다. `allowed_previous_resource_registries`
는 빈 객체여야 한다. 일부 프로세스만 새 등록부로 옮기는 순차 재연결은 사용하지 않는다.

예를 들어 GPU5 CPU producer가 새 PID로 바뀌었는데 GPU6이 이전 등록부를 계속 읽으면,
실제 사용 중인 producer RSS가 합산되지 않아 RAM 대기를 잘못 판단할 수 있다.
800GiB를 낮춰서 이 관측 누락을 숨기지 않는다.

## 실행 전 증거

도구에는 `--execute`, 새 BASE 직하의 실험 폴더, 그리고 다음 파일과 각 SHA256을 명시한다.

- 현재 7개 등록 행이 들어 있는 manifest.
- 정확한 이전 컨트롤러·native·DA worker·CPU producer·tracker의 종료 증거.
- 현재 CPU 후보 bank의 보존 snapshot을 연결하는 source updates.
- 이미 검토한 기존 G3 코드 export 증거.

Source updates는 다음 다섯 항목만 변경한다.
`previous_partial_banks`, `previous_producer_request`, `previous_producer_request_sha256`,
`interrupted_candidate_recovery_snapshot`, `interrupted_candidate_recovery_snapshot_sha256`.
전체 512/1024 후보 bank의 642개 source와 유효한 2048 결과를 보존한다.
새 `banks` 폴더는 producer가 생성하므로 미리 만들지 않는다.

이 도구는 프로세스에 신호를 보내지 않는다. 이전 프로세스가 살아 있거나 종료 증거가
부족하면 시작 전에 실패한다. 중단 작업은 별도로 정확한 PID·birth·UID·argv를 확인하고,
대상과 이유를 먼저 보고한 뒤 수행한다.

## 보존과 발행 순서

GPU0–4·6의 실제 latest와 BEST를 CPU에서 읽어 전체 모델·optimizer·GradScaler,
8개 logger 배열과 완료 epoch를 검증한다. 완료 epoch를 과거 resume 기록이나
진행 화면에서 추정하지 않는다. 실제 학습 모델 증거는 B2, 250 epoch, 128³ 입력,
105명 train, 26명 validation, DA worker 4 조건을 유지해야 한다.

기존 private native 런타임·전처리·체크포인트 경로를 재사용한다. 체크포인트를 복사하거나
코드에 새 hardlink를 만들지 않는다. 실제 학습은 재사용한 mutable native 출력에 이어서 쓴다.
이전 immutable request·native·로그를 새 폴더에 보존하고, 진행 로그에는 완료된 epoch
구간만 연결한다. 새 native metadata를 해당 request보다 먼저 발행한다.

7개 request를 모두 발행한 다음, 정확한 새 PID·birth·UID·argv와 request SHA를 담은
공통 등록부 하나를 발행한다. 새 manifest는 현재 7개 행과 동일한 등록부 경로를 기록한다.
JSON은 새 임시 파일에 기록하고 fsync 후 원자적으로 승격한다. 기존 metadata를 덮어쓰지 않는다.
기존 컨트롤러는 새 코드 export로 바뀌지 않는다. 실행 중인 코드의 inode·ctime을 바꾸는
추가 hardlink 생성도 하지 않는다.

## 설정과 검증 범위

Native별 RSS 96GiB, GPU3/4/5 후보 풀 384GiB, 전체 RSS 768GiB, 공용 여유 32GiB,
전체 admission 800GiB를 유지한다. CPU 후보 준비 worker 24와 기존 CPU affinity도 유지한다.
모델·CP 조건·병변 수·후보 수·physical B2·250 epoch·BEST 전체 26명 평가·UI 양식은 변경하지 않는다.

저장소 도구는 검토한 2026-10-14 G3 export를 명시적으로 제한한다. 다른 code export에
재사용할 때 이 제한을 임의로 우회하지 않고 검토된 새 source/manifest 증거를 준비한다.

정적 검사와 CPU 안전 검사 10개를 수행했다. 이 문서의 작성만으로 원격 복구,
전체 CUDA 학습 또는 전체 평가가 완료된 것은 아니다. 실제 서버 실행 상태와 완료 증거는
별도의 manifest·프로세스 신원·whole checkpoint·BEST26 결과로 기록해야 한다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 B2와 7개 실험을 유지한다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 실제 프로세스가 빠진 등록부의 합산 문제를 확인했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이 복구는 관측 누락을 수정한다.
- [x] 디버그 설정과 최종 설정을 분리했다. CPU 검사 데이터를 실제 학습 결과로 보고하지 않는다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [ ] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 이번 변경은 운영 도구이며 학습 경로를 수정하지 않는다. 새 원격 실행 검증은 별도 기록한다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
