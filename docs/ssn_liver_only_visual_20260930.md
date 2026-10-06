# 간 내부로 제한한 SSN 진단 그래프

사용자의 명시적인 간 외부 제외 요청을 적용했다. 기존 liver_66 SSN partition을 organ=(label==1 또는 label==2)와 교차시키고, 잘려서 분리된 각 6-connected component를 각각 슈퍼노드로 보존했다. 간 내부 모든 model voxel과 종양 voxel을 유지한다. 접촉 edge는 남은 voxel의 실제 면 접촉으로 다시 계산한다. 중심 표시점은 평균 위치가 영역 밖에 있을 가능성을 피하기 위해 해당 영역 내부 voxel 중 평균에 가장 가까운 위치다. 연결선은 접촉 관계를 표시하며 혈관 경로가 아니다.

| Record | Nodes 전→후 | Edges 전→후 | 남은 voxel | 간 밖 voxel | 종양 voxel 보존 |
|---|---:|---:|---:|---:|---:|
| liver_66:1 | 257→131 | 1148→502 | 52812 | 0 | 330/330 |
| liver_66:19 | 433→197 | 1594→635 | 70882 | 0 | 288/288 |

이 작업은 기존 SSN partition을 자르는 **시각화용 그래프 재구성**이다. 기존 CNN은 간 밖 CT도 들어 있던 patch에서 계산되었다. 따라서 이번 결과는 간 외부 정보의 영향을 제거한 CNN 인코더나 학습 모델을 완성했다는 뜻이 아니다. CNN 재인코딩, SSN 재학습, production 변경은 수행하지 않았다. 주석은 clipping/표시용으로 사용했으며 암 여부를 학습 입력으로 추가하지 않았다. 전체 간 그래프도 아니다.

검증: tools/check_ssn_masked_graph.py에서 불연속 영역 분리, 간 내부 coverage, 외부 제외, 실제 접촉 edge, 내부 marker, 빈 mask 거부 확인. 실제 두 record의 원본 좌표 재구성 오차 0, 내부 marker 검증. 브라우저 두 record, 종양 표시, 모바일 폭과 캡처 확인. 결과 work/ssn_liver_only_20260930/report.json, visual_checks.json.

## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프 감소는 사용자 요청인 간 외부 제외이며 임의 cap이 아니다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] 학습 batch/병렬화 설정 미변경. 기존 두 결과를 재사용했다.
- [x] CPU 좌표/마스크 계산만 실행. GPU 학습과 자원 설정 변경 없음.
- [x] OOM 발생 없음. 메모리 회피용 모델 축소 없음.
- [x] 진단 결과와 production 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 새 학습 모듈 없음. forward/loss/optimizer 변경 없음을 명시했다.
- [x] 실제 설정·변경·간 외부 CNN 영향 미해결을 기록했다.
- [x] 단위·화면 검사와 전체 학습·평가를 구분했다. 학습·평가 미실행.
