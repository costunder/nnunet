# 비교 실험 저장 실패

2026-10-08 사용자 서버 로그, 실행 revision `b6d852367c0050768a02ef9fa655e43db0f60fcd`를 확인했다. `selected`, GPU 3의 GPU 실행 교정 결과를 `gpu_execution.jsonl`에 저장할 때 운영체제가 `OSError: [Errno 122] Disk quota exceeded`를 반환했다. 예외 종료 과정에서 `progress.json`의 임시 파일도 같은 오류로 생성하지 못했다.

이는 저장 할당량에 의한 쓰기 거부라는 직접 증거다. GPU OOM이나 앞선 입력 병렬화의 수치 오류로 분류하지 않는다. 과거 `df`의 NFS 전체 가용량 3.2TB는 사용자·그룹·프로젝트의 quota 잔여량을 증명하지 않는다. 적용된 quota의 종류, byte/inode 제한, 실제 한도는 이 로그에 없으며 1TB 한도라고 추정하지 않는다.

실패 지점은 실제 optimizer update 전 교정 결과 저장이다. supervisor는 기존 `selected/checkpoint_latest.pt`가 존재한다고 기록했다. 중단된 batch가 새로 저장됐다는 뜻이나 checkpoint 내용을 재검증했다는 뜻은 아니다.

## 실행 변경

- 독립 실행기가 비싼 입력 준비·GPU 실행에 들어가기 전에 실제 출력 위치에서 작은 파일의 생성·쓰기·flush/fsync·rename을 확인한다. 검사에 쓰는 private 파일만 정리한다. 기존 파일을 지우거나 교체하지 않는다.
- GPU 교정 시작 기록을 먼저 저장한다. 현재 기록조차 저장할 수 없으면 교정 계산을 시작하지 않는다.
- 저장 실패를 무시하고 학습을 계속하지 않는다. 진행률 종료 중 추가 오류가 생기면 원래 오류를 유지하고 추가 실패를 함께 알린다.

작은 쓰기 검사는 **그 순간 해당 쓰기가 가능함**만 확인한다. 다음 121MiB checkpoint, 이후 캐시 증가 또는 다른 실험의 동시 쓰기를 위한 quota를 예약하지 않는다. 이 수정 자체로 서버 quota가 해제되거나 공간이 확보되지 않는다.

## 캐시와 공간 확보

현재 `preparation_reuse`는 완성된 immutable 캐시를 가져올 때 hardlink를 먼저 시도한다. 다른 파일시스템 또는 hardlink 권한 제한에 한해서 복사한다. 같은 파일시스템에서 hardlink에 성공한 파일은 경로가 여러 개여도 데이터 블록이 공유된다. 반대로 기존에 각 arm에서 별도로 생성한 파일을 자동으로 찾아 합치지는 않는다. `source_prepared`, `compact_upper_v1`, `sample_layout`의 별도 게시도 남는다. `sample_layout`은 큰 local graph와 dense patch를 참조하고 작은 상위 구조만 저장한다.

따라서 arm별 `du` 수치를 그대로 더해 전부 회수 가능한 중복 용량이라고 보고하지 않는다. 현재 실험의 캐시를 지우면 입력 재생성으로 느려질 수 있다.

이미 제공된 구형 그래프 캐시 조사에서는 다음 경로의 index 등록 sample 총 609개, 약 48.08GiB가 정리 후보였다. 실제 잔존 파일과 회수량은 서버에서 다시 확인해야 한다.

| `/home/aicompetition06/Medical/` 아래 | sample 수 | 이전 회수 추정 |
| --- | ---: | ---: |
| `HierCP/work/full/graphs` | 235 | 18.31GiB |
| `HierCP/work/paired_basic_vs_hiercp/folds/fold_0/gnn/graphs` | 187 | 14.79GiB |
| `HierCP/work/paired_basic_vs_hiercp/folds/fold_1/gnn/graphs` | 187 | 14.98GiB |

기존 `tools/retire_old_v1_graphs.py`는 옵션 없이 읽기 전용으로 해당 inventory와 잔존 크기를 검사한다. `--apply`는 삭제 동작이므로 정확한 대상에 대한 사용자 승인 및 그 구형 캐시를 사용하는 작업이 없는 조건이 필요하다. 파일 수·index·파일명·소유자·hardlink 수·stat 변경을 검사하고 등록된 sample만 개별 삭제한다. index, 모델, prototype, 결과, 원본 CT와 현재 네 실험 경로는 대상이 아니다. 이 문서는 서버 삭제 완료 기록이 아니다.

이번 대화에서 사용자는 **“삭제 명령 준비 — 해당 구형 작업 실행 중 아님”**으로 위 범위를 승인했다. 삭제 대상과 현재 실험의 경로를 더 넓히지 않는다. 이미 서버에 있는 `HierCP-run-b6d8523`의 정리 도구를 사용하므로 quota가 막힌 상태에서 새 worktree를 먼저 만들 필요가 없다. 정리 명령에는 학습 재개를 연결하지 않는다.

```bash
python -B -u /home/aicompetition06/Medical/HierCP-run-b6d8523/tools/retire_old_v1_graphs.py --apply
```

## 검증 범위

회귀 검사는 실제 저장 가능 검사와 EDQUOT/ENOSPC 등의 오류 주입, 기존 checkpoint 보존, GPU 교정 호출 이전 실패, 원래 예외 보존을 대상으로 한다. quota가 제한된 실제 서버 또는 서버 학습 재개 성공을 대신하지 않는다. 모델·입력·10mm·학습/평가 후보·40epoch·physical batch·Adam·resume cursor는 변경하지 않는다.

저장·GPU 교정 제어·진행률·실행기 회귀 검사 62개가 통과했다. Python 3.10 구문 검사 9개 파일도 통과했다. 코드 SHA와 검사 범위는 [검증 기록](../validation/comparison_storage.json)에 보존했다. 이번 수정 검증에서 실제 CT/CUDA 및 서버 학습은 실행하지 않았다.

기존 정리 도구의 보호 검사 7개를 새 UNIT 임시 파일로 다시 실행해 통과했다. 등록된 609개만 삭제, 재실행 시 이미 없는 파일 구분, index/sample 변경·공유 hardlink·잘못된 경로·모델 파일 삽입 거부, 현재 실험 파일 보존을 확인했다. 서버 삭제 수행이나 회수량 확인은 아니다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [ ] physical batch size와 병렬화 가능성을 새로 측정했다. (저장 실패 처리만 변경, 기존 설정 유지)
- [ ] GPU, CPU, RAM 활용 상태를 새로 측정했다. (이번 오류는 파일 저장 실패)
- [x] 모델 축소보다 실제 오류와 저장 경로를 먼저 조사했다.
- [x] 오류 주입 UNIT 검사와 production 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 학습 경로에 사용하지 않았다.
- [ ] forward, loss, gradient와 optimizer를 이번 변경에서 새로 실행했다. (모델 경로 변경 없음)
- [x] 실행 변경과 저장 공간 미해결 범위를 명확하게 보고했다.
- [x] 오류 처리 검사와 서버 전체 학습·전체 평가를 구분했다.
