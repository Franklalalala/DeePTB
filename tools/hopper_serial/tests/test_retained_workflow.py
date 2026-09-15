import os
from pathlib import Path
import shutil
import subprocess
import time
import pytest


@pytest.mark.skipif(not shutil.which('bash'), reason='requires bash')
@pytest.mark.parametrize('returncode', [0, 7])
def test_controller_exit_does_not_end_allocation(tmp_path, returncode):
    (tmp_path/'DEPLOY_READY.json').write_text('{}')
    (tmp_path/'controller.py').write_text('raise SystemExit('+str(returncode)+')\n')
    script=Path(__file__).resolve().parents[1]/'retained_workflow.pbs'
    with (tmp_path/'parent.log').open('w') as log:
        proc=subprocess.Popen(['bash',str(script)],env=dict(os.environ,FLOW_ROOT=str(tmp_path),FLOW_ARM='onsite'),
                              stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+10
            while 'WORKFLOW_EXIT' not in (tmp_path/'parent.log').read_text():
                assert proc.poll() is None
                assert time.monotonic()<deadline
                time.sleep(.1)
            assert 'rc='+str(returncode) in (tmp_path/'parent.log').read_text()
            assert proc.poll() is None
        finally:
            proc.terminate();proc.wait(timeout=10)
        assert proc.returncode==143
