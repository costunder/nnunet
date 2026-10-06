# SSN 그래프와 전체 CT의 좌표 대조

기존 SSN 진단은 liver_66의 48³ 국소 입력 두 개에 한정된다. 앞선 liver_108 간·종양 표시와 같은 케이스가 아니었다. 이번 화면은 liver_66으로 통일하여 전체 간 윤곽, 전체 종양 주석, 기존 그래프와 그 입력 범위를 동시에 표시한다. 전체 간을 덮는 SSN 그래프를 새로 구현하거나 생성한 결과가 아니다.

원본 CT/label SHA256과 affine 일치를 확인했다. 기존 canonical pos_mm/grid에서 native ROI 크기를 복원하고, 원본 CT의 crop·정규화·보간을 다시 수행했을 때 두 record 모두 저장된 target_patch와 최대 절대 오차 0이었다. SSN 배정 배열의 세 축은 실제 입력 배열의 세 축에 대응하도록 변환했다.

| Record | Native ROI shape | Native origin | Nodes | Undirected edges |
|---|---|---|---:|---:|
| liver_66:1 | 119×111×19 | 295,137,69 | 257 | 1148 |
| liver_66:19 | 119×111×19 | 372,170,65 | 433 | 1594 |

종양은 native voxel 간격(step=1)의 marching-cubes 표면이며 연결성분을 삭제하지 않았다. 간 윤곽은 모든 원본 단면을 사용하고 XY contour만 0.5 voxel 허용오차로 단순화했다. CT 단면 표시는 XY stride 3, Z stride 1이다. 모든 그래프 노드와 edge는 유지했다. 종양 라벨은 시각화와 node overlap 설명에만 사용하며 SSN 입력/학습을 바꾸지 않았다. overlap 비율은 종양 예측 확률이 아니다.

기존 SSN의 32-update CT 재구성 진단이라는 한계는 그대로다. 전체 간 SSN, CP 학습, 의미 있는 조직 분할 성능은 검증되지 않았다. 이번 작업은 새로운 학습을 하지 않았다.

검증 결과: work/ssn_whole_context_20260930/report.json 및 visual_checks.json. 브라우저에서 전체 node/edge 개수, 종양 토글, 두 record 전환, 회전, CT 단면, 노드 선택, 모바일 폭을 검사했다. whole.png를 시각적으로 확인했다.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다. 기존 국소 그래프 전부 표시.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다. 표시용 단면 간격 명시.
- [x] physical batch size와 병렬화 가능성을 검토했다. 기존 결과 재사용으로 모델 실행 불필요.
- [x] 자원 확인: 이번 작업은 CPU 파일 읽기·좌표 변환·표시이며 GPU 학습 없음.
- [x] OOM 발생 없음. 모델 축소 없음.
- [x] 디버그 설정과 최종 설정을 분리했다. production 설정 미변경.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 새 모델 모듈 없음. forward/loss/optimizer 연결 변경 없음.
- [x] 실제 표시 범위와 미구현 전체 간 그래프를 구분해 보고했다.
- [x] 화면 검사와 전체 학습·평가를 구분했다. 전체 학습·평가 미실행.
