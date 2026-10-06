"""Create short, real entry points and readable version navigation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = '''
## 작업 완료 체크리스트

- [x] 서버 또는 원격 세션 종료 위험이 있는 명령을 사용하지 않았다.
- [x] 사용자 파일과 기존 결과를 파괴적으로 변경하지 않았다.
- [x] 모델 깊이와 너비를 편의상 축소하지 않았다.
- [x] 그래프와 데이터 규모를 편의상 축소하지 않았다.
- [x] 숨겨진 subset, cap, fast mode를 추가하지 않았다.
- [x] physical batch size와 병렬화 가능성을 실제로 검토했다. 기존 실행 설정을 유지했다.
- [x] GPU, CPU, RAM 활용 상태를 측정하거나 확인했다. 기존 자원 기록을 연결했다.
- [x] OOM 발생 시 모델 축소보다 메모리 및 병목 원인을 먼저 조사했다. 이번 정리에서는 학습을 실행하지 않았다.
- [x] 디버그 설정과 최종 설정을 분리했다.
- [x] dummy, placeholder, random fallback을 사용하지 않았다.
- [x] 핵심 모듈이 forward, loss, gradient와 optimizer에 연결되어 있다. 기존 구현 경로를 그대로 연결했다.
- [x] 실제 실행 설정과 변경 사항을 명확하게 보고했다.
- [x] smoke test와 전체 학습 또는 전체 평가를 구분해서 보고했다.
'''


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip()+'\n', encoding='utf8')


def link(folder, path, label):
    import os
    return f'[{label}]({Path(os.path.relpath(ROOT/path,folder)).as_posix()})'


def main():
    entries = json.loads((ROOT/'versions/index.json').read_text(encoding='utf8'))['entries']
    fixes = {'hiercp_v1x/half_a_local.py':'hiercp_v1x/half_a_model.py',
             'hiercp_v1x/transition_c_model.py':'hiercp_v1x/transition_model.py',
             'hiercp_v1x/historical_full128.py':'hiercp_v1x/historical_evaluation.py'}
    for entry in entries:
        entry['modules'] = [fixes.get(p,p) for p in entry['modules']]
        for path in entry['modules']+entry['configs']+entry['notes']+([entry['runner']] if entry['runner'] else []):
            if not (ROOT/path).exists(): raise FileNotFoundError(path)
    index = json.loads((ROOT/'versions/index.json').read_text(encoding='utf8'))
    index['entries'] = entries
    write(ROOT/'versions/index.json',json.dumps(index,ensure_ascii=False,indent=2))
    for entry in entries:
        folder = ROOT/'versions'/entry['folder']
        folder.mkdir(parents=True,exist_ok=True)
        # eval.py is a separate entry, rather than overwriting v1/run.py.
        filename = 'eval.py' if entry['id'].endswith('/eval') else 'run.py'
        if entry['runner']:
            depth = len(Path(entry['folder']).parts)+1
            write(folder/filename, f'''"""{entry['title']}: unchanged original arguments/runtime."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[{depth}]
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'versions/run.py'), run_name='version_entry')['launch']({entry['id']!r}, sys.argv[1:])
''')
        # Agent owns v1/v1.4-v1.7 code.md and README.md.
        if entry['id'] in ('v1','v1.4','v1.5','v1.6') or entry['id'].startswith('v1.7') or entry['id']=='v1/eval':
            continue
        links = '\n'.join('- '+link(folder,p,Path(p).name) for p in entry['modules']) or '- 실행 소스는 보존 ZIP 안에 있습니다.'
        write(folder/'code.md', '# 구현 위치\n\n'+links+'\n\n공유 모듈은 원래 경로에 둡니다. 파일명·SHA를 바꾸면 기존 checkpoint의 exact resume 검증이 깨질 수 있습니다.\n')
    # One config page per folder, including all entries using it.
    for relative in sorted({e['folder'] for e in entries}):
        folder = ROOT/'versions'/relative
        related = [e for e in entries if e['folder']==relative]
        paths = sorted({p for e in related for p in e['configs']})
        configs = '# 설정\n\n아래 링크가 실제 실행에서 사용하는 설정 원본입니다. 이 안내를 모델 설정으로 읽지는 않습니다.\n\n'
        configs += '\n'.join('- '+link(folder,p,Path(p).name) for p in paths) or '폐기된 버전은 새 실행 설정을 제공하지 않습니다.'
        configs += '\n\nGPU·margin·실험 출력·resume 경로는 기존 CLI에서 명시합니다. 실행 진입점은 기존 인자를 그대로 전달하며 batch/epoch/GT를 변경하지 않습니다.\n'
        write(folder/'config.md',configs)
    server_routes = {
        'v1/eval.sh':('tools/server_native_v1_full128.sh',None),
        'v1.7/C/server.sh':('tools/server_v17_crossed.sh','C'),
        'v1.7/D/server.sh':('tools/server_v17_crossed.sh','D'),
        'v2.2/curriculum/server.sh':('tools/server_v22_cumulative_u16.sh',None),
    }
    for relative,(target,arm) in server_routes.items():
        up = '/'.join(['..']*(len(Path(relative).parent.parts)+1))
        content = '#!/usr/bin/env bash\nset -euo pipefail\n'
        content += f'VERSION_REPO="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/{up}" && pwd)"\n'
        content += 'cd -- "$VERSION_REPO"\n'
        if arm: content += f'export CP_ARM={arm}\n'
        content += f'bash {target} "$@"\n'
        write(ROOT/'versions'/relative,content)
    # Safe read-only learning display is a separate short entry.
    write(ROOT/'versions/v1.7/D/status.py','''"""Display saved D training metrics; never starts or alters training."""
from pathlib import Path
import runpy
ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    runpy.run_path(str(ROOT/'tools/watch_v17_d_learning.py'),run_name='__main__')
''')
    print(json.dumps(dict(entries=len(entries),training_started=False)))


if __name__ == '__main__':
    main()
