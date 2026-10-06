"""Readable short guides; retain detailed historical documents as evidence."""
from __future__ import annotations

import json
from pathlib import Path

from build_version_layout import ROOT, CHECKLIST, link, write

GUIDES = {
    'v2': ('폐기된 view-only 방향', '동일 데이터의 두 view를 정렬한 초기 방향입니다. 이후 실제 환자 간 정렬로 방향을 바꿨습니다. 현재 실행 버전으로 선택하지 않습니다.'),
    'v2.1': ('환자 간 prompt 정렬', 'v1 국소 표현을 바탕으로 환자별 label/data 관계와 환자 간 정렬을 도입했습니다. 내부 패키지 이름은 hiercp_v2이며 사람용 버전 번호 v2.1과 구분합니다.'),
    'v2.2/early': ('초기 관측·관계 그래프', '초기 관측 특징, CT-only 변경, 관측 관계·군집 정렬의 보존 구현입니다. run_v22 / run_v221 / run_v222의 format ID는 그대로 보존합니다. 현재 LocalCNN과 동일 모델이라고 부르지 않습니다.'),
    'v2.2/paired': ('v1 L0 paired 그래프', 'v1 형태의 CT 문맥 그래프 L0를 v2.2 상위 모델에 연결한 paired 준비·학습 경로입니다. 원본 v1 checkpoint나 현재 국소CNN 실험과 다른 artifact입니다.'),
    'v2.2/cnn': ('국소3D CNN L0', '간 mask 안의 native-spacing CT를 donor 종양 bbox 바깥 margin 범위로 읽습니다. CNN12/24/32 → scale별 masked readout → donor/recipient fusion128D → 기존 L1/L2로 이어집니다. margin은 각 면 바깥 여유거리이며 전체 cube 너비가 아닙니다. 기존 서버 학습에서 후보 순위 분리가 약했습니다.'),
    'v2.2/curriculum': ('CNN10mm + 누적 U16→128', 'LocalCNN과 전체 P/U 정의를 유지합니다. TRAIN은 모든 P+16U로 시작하고, 누적 pair-win≥70% 및 평균 bestP−bestU>0이 2epoch 연속이면 다음16U를 추가합니다. 이전U는 유지합니다. support와21환자 validation/BEST는 처음부터 전체128U입니다. 40epoch 안에128까지 가지 못하면 미완료 노출을 기록합니다. 장기 성능 결과는 아직 받지 않았습니다.'),
    'v2.2/regions': ('고정 영역 축약 + SAGE', '고정 특징으로 partition·coarse edge를 준비한 뒤 최신 CNN 특징을 영역별로 집계하고 SAGE를 수행합니다. 현재 partition의 bbox·분산 등 초기 품질 기준 위반과 의미 혼합 문제가 미해결입니다. 준비 성공이나 노드 감소를 추천 품질로 간주하지 않습니다.'),
    'v2.2/sage': ('축약 없는 SAGE', '기존 fine sampled graph를 유지하고 GAT 대신 SAGE로 메시지를 전달합니다. 과거 동일 update 비교의 속도 개선은 전체 epoch나 추천 정확도 개선을 보장하지 않습니다. runner는 저장된 checkpoint에서 기존 설정을 읽고 그래프 전환을 명시합니다.'),
    'v2.2/sparse': ('작은 문맥 그래프', 'CNN 위치별 특징과 역할·shell·상대 위치를 작은 그래프로 보존하는 연구 경로입니다. footprint/relay 등 대안과 실제 CP 실행 DEBUG가 있습니다. 별도 환자 순위 개선은 입증되지 않았습니다. 일부 최신 연구 소스는 로컬 미게시 파일이며 production 실행기로 표시하지 않습니다.'),
    'v2.2/explore': ('좌표 탐색 그래프', 'CNN 특징을 읽는 연속 좌표와 parent→child 탐색 연결을 시험한 DEBUG 경로입니다. 초기 구현의 중심 집중·중복 위치와 문맥 범위 문제가 보고됐습니다. 생성된 node/edge나 gradient 유무를 의미 있는 탐색의 증거로 쓰지 않습니다. 일부 진단 실행기는 로컬 미게시 파일입니다.'),
    'v2.2/control': ('별도 연구 대조', 'CNN-only, GraphUNet, SSN 시각화 등 별도 연구·DEBUG 기록입니다. 현재 서버 학습과 동일 모델이 아니며 종양·혈관 의미 분할이 검증됐다고 표시하지 않습니다.'),
    'common': ('공유 기능', 'Basic CP, nnU-Net·feedback, validation/assembly, 데이터·자원·저장소 도구는 여러 버전이 공유합니다. 각 버전 번호를 붙여 복제하지 않습니다.'),
}


def main():
    index = json.loads((ROOT/'versions/index.json').read_text(encoding='utf8'))
    for relative, (title, explanation) in GUIDES.items():
        folder = ROOT/'versions'/relative
        related = [e for e in index['entries'] if e['folder']==relative]
        content = f'# {title}\n\n{explanation}\n\n'
        content += '| 찾을 것 | 파일 |\n| --- | --- |\n'
        if any(e['runner'] for e in related): content += '| 실행 | [run.py](run.py) — 기존 CLI 인자 그대로 |\n'
        content += '| 설정 원본 | [config.md](config.md) |\n| 구현 | [code.md](code.md) |\n'
        if (folder/'server.sh').exists(): content += '| 서버 진입점 | [server.sh](server.sh) — 기존 CP_GPU/출력 설정 유지 |\n'
        content += '\n'
        notes = sorted({p for e in related for p in e['notes']})
        if notes:
            content += '세부 기록: '+', '.join(link(folder,p,f'기록{i+1}') for i,p in enumerate(notes))+'.\n\n'
        if relative not in ('v2','v2.1','common'):
            content += 'P는 실제 관측된 적격 종양 위치이고 U는 미관측 비교 위치입니다. CP suitability 정답으로 재정의하지 않습니다.\n\n'
        content += CHECKLIST
        write(folder/'README.md',content)
    previous = ROOT/'versions/README.md'
    old = ROOT/'versions/history.md'
    if previous.exists() and not old.exists():
        # Preserve previous explanatory text byte for byte before replacing navigation.
        old.write_bytes(previous.read_bytes())
    write(previous, '''# 버전별 작업 폴더

현재 안내는 [START](../START.md)입니다. 긴 날짜·hash를 붙인 새 작업 폴더를 만들지 않고 아래 버전과 방법 이름을 사용합니다.

| 버전 | 실험 내용 | 안내 |
| --- | --- | --- |
| v1 | 기존 그래프, 원본30mm와 보존 source의 차이 | [열기](v1/README.md) |
| v1.4 | 10mm 기준선 | [열기](v1.4/README.md) |
| v1.5 | A: L0 교체 | [열기](v1.5/README.md) |
| v1.6 | B: 상위 계층·scorer 교체 | [열기](v1.6/README.md) |
| v1.7 | C/D 교차 실험 + 공통 전체128 평가 | [열기](v1.7/README.md) |
| v2 | 폐기한 view-only 방향 | [열기](v2/README.md) |
| v2.1 | 환자 간 prompt 정렬 | [열기](v2.1/README.md) |
| v2.2 | 현재 방법별 구현 | [열기](v2.2/README.md) |
| common | Basic CP·nnU-Net·공유 기능 | [열기](common/README.md) |

각 폴더는 `README.md / run.py / config.md / code.md / results.json`처럼 짧게 읽습니다.
방법이 여러 개면 같은 버전 아래 `cnn / curriculum / sage / regions / sparse / explore`로 구분합니다.
폐기·미검증 방법은 상태를 표시하고 자동으로 학습시키지 않습니다.

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
데이터 cache·결과·환경을 이름만 보고 더미라고 삭제하지 않습니다.
'''+CHECKLIST)
    write(ROOT/'versions/v2.2/README.md', '''# v2.2

여러 L0 방법과 학습 수정이 같은 연구 계열 안에 있습니다. 실행 방법을 아래 폴더로 구분합니다.

| 폴더 | L0·학습 | 상태 |
| --- | --- | --- |
| [cnn](cnn/README.md) | 국소3D CNN | 기존 서버 순위 학습 약함 |
| [curriculum](curriculum/README.md) | CNN10mm, U16→128 누적 학습 | 구현/CUDA smoke 완료, 장기 결과 대기 |
| [sage](sage/README.md) | fine graph 유지 + SAGE | 기존 전환 경로 |
| [regions](regions/README.md) | 고정 partition + coarse SAGE | 영역 품질 미해결 |
| [sparse](sparse/README.md) | CNN 특징 + 작은 역할·문맥 그래프 | DEBUG, 추천 품질 미해결 |
| [explore](explore/README.md) | 연속 좌표 탐색 그래프 | DEBUG, 탐색 타당성 미해결 |
| [paired](paired/README.md) | v1 형태의 paired L0 | 과거 구현 |
| [early](early/README.md) | 초기 관측·관계 모델 | 과거 구현 |
| [control](control/README.md) | CNN/GraphUNet/SSN 대조 | 별도 연구·진단 |
| [history](history/README.md) | 변경 전 소스·검증 보존본 | byte-preserving archive |

각 폴더의 run.py가 실제 원래 실행기를 호출합니다. 기존 import package는 공유 runtime로 유지합니다.
`hiercp_v22 / hiercp_v221 / hiercp_v222`라는 파일 경로와 내부 format ID는 모델·artifact 검증 계약입니다.
사람용 버전/방법 폴더와 혼동하지 않습니다.

현재 선택한 새 실험은 **curriculum: 국소 CNN10mm + 누적 U16**입니다.
원래 D는 [v1.7/D](../v1.7/D/README.md)의 v1 그래프 L0 실험입니다.
모든128 후보를 평가한다는 사실만으로 두 모델을 같은 버전 설정이라고 보지 않습니다.

세부 소스는 [files.md](files.md), 결과 폴더는 [runs.md](runs.md)에서 찾습니다.
'''+CHECKLIST)
    print(json.dumps(dict(guides=len(GUIDES)+2,training_started=False)))


if __name__ == '__main__':
    main()
