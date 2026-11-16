"""Guard authorized scope and pilot-before-formal ordering without training."""
import contextlib
import io
from local_ablation_queue import arguments,build_plan,ABLATIONS


def main():
    plan = build_plan(arguments([]))
    assert plan['main_experiments_included'] is False
    assert plan['formal_experiments']==6 and plan['full_fold_models']==30
    assert len(plan['jobs'])==12
    assert [job['phase'] for job in plan['jobs']]==['pilot']*6+['full']*6
    for job in plan['jobs']:
        command=job['command']
        assert job['variant'] in ABLATIONS and 'none' not in command
        assert command[command.index('--ablation')+1]==job['variant']
        assert '--checkpoint-bert' in command and '--no-refresh-report' in command
        assert command[command.index('--algorithm-profile')+1]=='paper_algorithm_v4'
        if job['phase']=='pilot':
            assert command[command.index('--epochs')+1]=='1'
            assert '--limit-train' in command and '--limit-eval' in command
        else:
            assert command[command.index('--epochs')+1]=='30'
            start=command.index('--folds')+1
            assert command[start:start+5]==['1','2','3','4','5']
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            arguments(['--variants','none'])
        except SystemExit as error:
            assert error.code!=0
        else:
            raise AssertionError('Main model allowed into ablation queue')
    assert len(build_plan(arguments(['--pilots-only']))['jobs'])==6
    print('PASS: six ablation pilots precede six formal runs; 30 fold models; main training excluded')


if __name__=='__main__':
    main()
