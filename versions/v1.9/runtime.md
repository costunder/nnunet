# 실행 비용과 캐시 정리

2026-10-07 서버 로그와 실행 코드를 대조했다. 원래 10mm 학습이 약 30분이었다는 사용자 기록을 기준으로, 현재 실행의 추가 비용을 확인했다. 모델·10mm 범위·학습 후보 8개를 줄이는 수정은 적용하지 않았다.

## 원래 실행과 달라진 작업

| 작업 | 원래 v1 m10 | 현재 네 군 |
| --- | --- | --- |
| 학습 | 151문제 × 8후보 | 동일 |
| 평가 | 36문제 × 8후보 | 36문제 × 129후보 |
| 입력 | 준비된 sample 파일을 mmap·DataLoader로 읽음 | CT·fields·canonical·upper를 조립하고 두 view 생성 |
| 저장 | epoch 끝 | update 및 validation batch마다 저장 |

평가 후보 수는 288에서 4,644로 늘었다. 실행시간이 16.125배가 된다는 뜻은 아니다. 현재 약 121MiB checkpoint를 151 update와 36 validation batch마다 다시 쓰면 약 22GiB/epoch의 쓰기 I/O가 생긴다. 저장공간에 22GiB가 누적된다는 뜻은 아니다. 저장 빈도와 전체 평가 계약은 이번 수정에서 유지했다.

현재 첫 사용에는 raw CT·label의 SHA·디코딩, whole-case 거리 배열 검증, 없는 후보 그래프 생성이 들어간다. `cache_miss.wall_seconds`는 다른 군의 완성 캐시를 찾는 시간이며, 그 뒤 실제 생성 비용은 포함하지 않는다. 순환 후보 군은 다음 epoch에도 새로운 U의 canonical 그래프를 만들 수 있다. 고정 후보 군은 만들어진 canonical을 다시 사용할 수 있지만, 원래 두 sampled view와 collate는 계속 수행한다.

## 제공된 로그에서 실제로 측정된 것

| 구간 | 전체 경과 | update의 `sec` |
| --- | --- | --- |
| native_fixed 34→35 | 약 14초 | 5.24초 |
| native_listwise 34→35 | 약 11초 | 5.24초 |

이 구간의 `55~74초/it`는 이전 느린 batch가 섞인 tqdm 추정치다. 현재 batch의 경과 시간과 같지 않다. `sec`에는 입력 전송·forward·backward·검사·optimizer·checkpoint 저장이 포함되며 loader wait은 제외된다.

402MB의 liver_55 fields를 처음 다시 열 때 fixed는 11.140초, listwise는 10.279초가 걸렸다. `reopened`는 거리 변환을 다시 계산하지 않았다는 뜻이다. 기존 파일의 전체 SHA와 수치 검사는 수행한다. 같은 process에서 검증된 mapping은 이후 재사용한다. 모든 문제마다 fields 전체를 다시 해시한다고 해석하면 틀린다. worker와 prefetch에서 겹친 시간을 합산해 병목 비중으로 제출하지 않는다.

## 이번 수정

- RAM에 있는 canonical graph는 원래 `_get`의 hit 경로를 먼저 사용한다. 추가 파일 경로 조회와 publication guard는 RAM miss에서만 수행한다.
- RAM에 있는 static upper와 살아 있는 동일 binding의 fields mapping도 기존 loader의 hit 경로로 읽는다. 원래 budget·LRU·방어적 복사·receipt를 유지한다.
- 최초 load/import의 SHA·binding·수치 검사와 기존 캐시 게시 보호는 유지한다. 검증을 생략하거나 가짜 cache hit를 만들지 않는다.
- 개별 캐시 JSON과 원본 fields 출력은 `preparation_reuse.jsonl`에 보존한다. 터미널에는 60초마다 짧은 집계, 종료 집계와 오류·메모리 압력만 표시한다. 다른 진행률·오류는 그대로다.

이는 추가 I/O와 콘솔 출력을 줄이는 실행 수정이다. 최초 raw/fields 검사, 새 후보 생성, 전체129 평가와 checkpoint 쓰기는 남는다. 서버의 warm epoch 시간이나 30분 복귀는 아직 측정되지 않았다. 실행 중인 process에 새 코드가 자동 반영되지 않으며, 다음 재개부터 적용한다. checkpoint와 네 군의 변인은 유지한다.

## 승인된 과거 캐시만 정리

사용자가 확인한 세 과거 경로의 sample 캐시 609개만 대상으로 한다.

| 경로: `/home/aicompetition06/Medical/` 아래 | 파일 수 | 회수 추정 |
| --- | ---: | ---: |
| HierCP/work/full/graphs | 235 | 18.31GiB |
| HierCP/work/paired_basic_vs_hiercp/folds/fold_0/gnn/graphs | 187 | 14.79GiB |
| HierCP/work/paired_basic_vs_hiercp/folds/fold_1/gnn/graphs | 187 | 14.98GiB |

합계는 약 48.08GiB다. 사용자 확인에서 shared hardlink는 없었고, 현재 실행의 JSON 21개에 세 경로 참조가 발견되지 않았다. 이 값은 allocated block 기준 추정이며 NFS quota 한도나 실제 회수 완료를 증명하지 않는다.

`tools/retire_old_v1_graphs.py --apply`는 세 index의 정확한 파일 수·case/sample filename·regular file·소유자·nlink=1을 검사한다. 전체 사전검사와 개별 unlink 직전 stat 검사를 수행하며, 다른 파일과 디렉터리는 삭제하지 않는다. index가 변경되거나 파일이 교체되면 중단한다. 세 은퇴 캐시에 writer가 없다는 조건에서 사용하며, 비협조적 동시 writer와 stat→unlink 사이를 원자적으로 격리한다고 주장하지 않는다.

모델·prototype·설정·결과·원본 영상·index·현재 네 군의 data와 v18 원본 source를 보존한다. 삭제 목록을 시스템 임시 폴더에 먼저 기록하며, 부분 삭제 후 같은 명령은 이미 없는 indexed 파일을 따로 집계한다. 과거 세 실험의 graph cache를 다시 쓰려면 sample을 재생성해야 한다. 서버 삭제는 사용자가 명령을 실행한 뒤 결과로 확인한다.

## 검증 범위

집중 UNIT 검사: cache hot path 및 기존 reuse/fields/upper/host-memory 85개, 출력·재개 30개, 삭제 보호 7개 PASS. 삭제 검사는 새 UNIT 임시 파일에서만 수행했다. 모델·서버 결과·기존 캐시는 삭제하지 않았다. 실제 CT/CUDA 회귀의 원본 모델·gradient·update·전체129 평가·재개 결과와 코드 SHA는 [실행 기록](../../validation/cache_runtime/execution.json)에 별도 보존한다. 로컬 실행 통과를 서버 시간 개선이나 추천 품질 개선으로 해석하지 않는다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다. 승인된 과거 cache 정리 도구는 아직 서버에서 실행하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 측정 lock과 worker를 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 서버 로그와 별도 실제 CT/CUDA DEBUG를 구분한다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 변경은 추가 파일 접근과 로그 출력이다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 원래 엔진을 보존하며 별도 CUDA 회귀로 확인한다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다. 서버40epoch·시간 개선·추천 품질은 이번 수정의 검증 범위가 아니다.
