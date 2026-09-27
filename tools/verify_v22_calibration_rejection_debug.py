"""Actual process admission of deliberately corrupted DEBUG calibration copies."""
from pathlib import Path
import argparse,copy,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cache',type=Path);p.add_argument('calibration',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=('training_started.json','loader_calibration.json','memory_batch_calibration.json','training_batch_calibration.json')
    source={name:json.loads((a.calibration/name).read_text(encoding='utf-8')) for name in names}
    results=[]
    for variant in ('CE','legacy','runtime','dict','empty','unmeasured_batch','NaN'):
        folder=a.output/variant;folder.mkdir();reports=copy.deepcopy(source);old=reports['training_started.json']
        if variant=='CE':old['training_objective']='observation_ce'
        elif variant=='legacy':old['feature_coordinates']='legacy'
        elif variant=='runtime':old['calibration_runtime_sha256']={}
        elif variant=='dict':reports['training_batch_calibration.json']={}
        elif variant=='empty':reports['memory_batch_calibration.json']=[]
        elif variant=='unmeasured_batch':old['physical_batch']=9999
        elif variant=='NaN':reports['training_batch_calibration.json'][0]['graphs_per_second']=float('nan')
        for name,value in reports.items():(folder/name).write_text(json.dumps(value),encoding='utf-8')
        command=[sys.executable,'-u',str(ROOT/'tools/run_v222_process_runtime.py'),'--debug-profile','review_repair',
            '--cache',str(a.cache.resolve()),'--calibration',str(folder.resolve()),'--output',str((folder/'run').resolve()),
            '--workspace-mib','256','--release-unused']
        with (folder/'console.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        lines=(folder/'console.log').read_text(encoding='utf-8').splitlines();error=lines[-1] if lines else ''
        if result.returncode==0 or not error.startswith('ValueError:') or (folder/'run/checkpoint_latest.pt').exists():
            raise AssertionError((variant,result.returncode,error))
        results.append(dict(variant=variant,rejected=True,before_first_checkpoint=True,error=error))
        print(json.dumps(results[-1]),flush=True)
    (a.output/'result.json').write_text(json.dumps(dict(debug=True,actual_process=True,cases=results),indent=2),encoding='utf-8')


if __name__=='__main__':main()
