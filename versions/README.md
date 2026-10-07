# 버전별 작업 폴더

파이프라인 요약은 [루트 README](../README.md), 버전·방법 선택은 이 표를 사용합니다. 소스와 실험 결과는 기존 경로를 유지하며 여기서는 짧은 안내 폴더로 연결합니다.

| 분류 | 버전·방법 | 확인된 상태 |
| --- | --- | --- |
| 기준선 | [v1 원본30mm](v1/README.md), [v1.4 10mm](v1.4/README.md) | 원본30mm와 보존 ZIP의 소스 시점을 구분한다. V1.4 자체8후보 학습 성공은 사용자 서버 기록이며 CP 추천 품질 증거가 아니다. |
| 현재 네 군 비교 | v1.8 [selected](v1.8/selected/README.md) · [native](v1.8/native/README.md), v1.9 [fixed](v1.9/fixed/README.md) · [listwise](v1.9/listwise/README.md) | 같은 원본10mm 모델의 좌표 노출·loss 대조. 실제 CT/CUDA DEBUG·재개 검증과 전체40epoch 완료·최종 품질을 구분한다. 서버 live 상태는 별도 확인한다. |
| 기존 원인 분리 | [A: v1.5](v1.5/README.md), [B: v1.6](v1.6/README.md), [C/D: v1.7](v1.7/README.md) | A/B/C 자체8후보 학습 완료 보고가 있다. D 최종 완료·평가는 미확인이다. |
| 별도 실험 | v2.2 [cnn](v2.2/cnn/README.md) · [curriculum](v2.2/curriculum/README.md) · [sage](v2.2/sage/README.md) · [regions](v2.2/regions/README.md) · [sparse](v2.2/sparse/README.md) · [explore](v2.2/explore/README.md) | 방법별 한계와 DEBUG 범위를 확인한다. curriculum은 서버23epoch 보고, U16 gate 미통과, 품질 미검증이다. |
| 이전 방향·보존 | [v2 view-only](v2/README.md), [v2.1 환자 간 정렬](v2.1/README.md), [v2.2 history](v2.2/history/README.md) | 폐기·과거 구현을 현재 검증된 모델로 표시하지 않는다. |
| 공유 기능 | [common](common/README.md) | Basic CP·nnU-Net·feedback·공유 도구 |

전체 조건은 [v1.8](v1.8/README.md)·[v1.9](v1.9/README.md), 다른 V2.2 방법은 [v2.2 목록](v2.2/README.md)을 읽습니다. 네 군의 짧은 폴더에는 설명과 `server.sh`가 있으며, 기존 GPU·실험 환경변수를 명시해 군 하나를 재개합니다.

버전 폴더의 `README.md / run.py / config.md / code.md / results.json`은 해당 파일이 있는 곳에서 사용합니다. 8후보 source-anchor 점수와 전체 P+128U 점수는 다른 평가이며, P/U는 CP 적합·부적합 정답이 아닙니다.

Vx.yz에서 x는 방향 전환, y는 의미 있는 개선, z는 버그 수정입니다.
버그 패치마다 최상위 폴더를 만들지 않습니다. 과거 `v2.21`, `v2.22`라는 저장 이름과 내부 format ID는
[v2.2/history](v2.2/history/README.md)에 보존하며 새로운 의미의 버전 번호를 덧씌우지 않습니다.
v1.1~v1.3은 이전 계획에 있던 단계이며 실행 완료된 모델로 만들지 않습니다.

실제 실행 목록:

```bash
python versions/run.py list
python versions/v2.2/cnn/run.py --help
python versions/v1.7/C/run.py --help
```

원래 인자를 그대로 사용합니다. GPU·margin·출력·checkpoint를 자동 교체하지 않습니다.
실행 코드와 설정은 원본 파일명·SHA를 유지하고, 새 run.py가 그 파일을 실제 호출합니다.
각 버전의 [files.md](v2.2/files.md)에서 구현·설정·도구·검사·기록을 찾습니다.
[실험 위치](artifacts.md)는 기존 cache/checkpoint 경로를 버전별로 묶어 보여줍니다.
보존된 생성 fixture는 묶음별 개수와 원래→현재 이동 receipt로 찾습니다. 묶음 수를 학습 실험 수로 세지 않습니다. 데이터 cache·결과·환경을 이름만 보고 더미라고 삭제하지 않습니다.

## 작업 완료 체크리스트

이번 변경은 탐색 문서·인덱서다. 새 학습·평가·자원 benchmark는 실행하지 않았다.

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토한 기존 실행 기록을 연결했다. 새 측정은 없다.
- [x] GPU, CPU, RAM 활용 상태를 측정한 기존 자원 기록을 연결했다. 새 측정은 없다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 조사한 기존 기록을 연결했다. 이번 정리에는 OOM이 없다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈의 forward, loss, gradient와 optimizer 연결에 관한 기존 검증 범위를 구분했다. 새 실행 검사는 없다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
