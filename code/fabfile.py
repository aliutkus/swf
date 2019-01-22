from fabric.api import hosts, run, env, get, cd, task
import uuid
import os
import itertools as it
import json
import time

env.output_prefix = False

cluster = 'fstoter@nef-frontal.inria.fr'


# project folders on nef
DATA_PATH = '/local/read'
PATH_PROJECT = '~/open-unmix/pytorch'
PATH_JOBS = '~/jobs'


def oar_zenith(settings):
    job_string = """#!/bin/bash -l
#
#OAR -l /gpunum=1,walltime=20:00:00
#OAR -p dedicated='"'"'zenith'"'"' and gpucapability='"'"'6.1'"'"' and gpu='"'"'YES'"'"'
#OAR --notify mail:fabian-robert.stoter@inria.fr
#OAR -n {job_name}

echo "Hello from `hostname`"

module load cuda/9.1
source activate SWF

cd ~/open-unmix/pytorch

python swf.py \\
    MNIST \\
    --root_data_dir /local/read/fstoter/data/MNIST \\
    --img_size 32 \\
    --num_sketches 1 \\
    --clip 3000 \\
    --num_quantiles 100 \\
    --input_dim 32  \\
    --num_samples 3000 \\
    --particles_type RANDOM \\
    --stepsize 4000 \\
    --num_thetas 16000 \\
    --logdir ~/logs/MNIST \\
    --plot_every -1 \\
    --bottleneck_size {bottlneck_size} \\
    --ae_model ae \\
    --num_test 3000 \\
    --test_type RANDOM

""".format(**settings)

    return job_string


def oar_cluster_test(settings):
    job_string = """#!/bin/bash -l

#OAR -n {job_name}
#OAR -l /core=1,walltime=00:01:00
#OAR --notify mail:fabian-robert.stoter@inria.fr

echo "Hello from `hostname`"
echo "Path is : $PATH"
""".format(**settings)

    return job_string


def run_oar(oar, oar_name=None):
    if oar_name is None:
        oar_name = uuid.uuid4().hex[:6]

    oar_file = os.path.join(PATH_JOBS, oar_name + ".oar")
    with cd(PATH_PROJECT):
        run(
            'echo \'%s\' > %s; chmod +x %s' % (oar, oar_file, oar_file),
            shell=False,
            quiet=True
        )
        run('oarsub -S %s' % oar_file, shell=True)


@hosts(cluster)
@task
def start_experiment(experiment):

    with open(experiment) as exp_json_file:
        settings = json.load(exp_json_file)

    for setting in dproduct(settings):
        # use unique job names so it's easier to find currently running jobs
        uuids = uuid.uuid4().hex[:6]

        setting['job_name'] = setting['exp'] + "_" + uuids + "_" + target
        setting['uuid'] = uuids

        # add variables
        setting['output'] = "~/results/" + setting['exp'] + "_" + uuids

        cur_pbs = oar_zenith(setting)
        run_oar(cur_pbs, setting['job_name'])
        time.sleep(1)


@hosts(cluster)
@task
def start_cluster_test():
    run_oar(oar_cluster_test({'job_name': 'cluster_test'}))


@hosts(cluster)
@task
def uj():
    """Get the current user jobs."""
    run('oarstat -u')


@hosts(cluster)
@task
def del_all_jobs():
    """Delete all jobs of current user from queue"""
    run('''oarstat -u | tail -n+3 | gawk '{print $1}' | xargs oardel''')


@hosts(cluster)
@task
def deploy_head():
    """Go to the project directory and pull the latest version"""
    with cd(PATH_PROJECT):
        run('git pull origin dev')


@hosts(cluster)
@task
def clean_logs():
    """Delete log/oar files."""

    #  delete job files
    with cd(PATH_JOBS):
        run('rm *.oar')
    # delete o/e files
    with cd(PATH_PROJECT):
        run('rm OAR*.stderr*')
        run('rm OAR*.stdout*')


def dproduct(dicts):
    return (dict(zip(dicts, x)) for x in it.product(*dicts.values()))
