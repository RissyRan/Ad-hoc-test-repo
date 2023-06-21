"""DAG template for creating model tests."""

from airflow.models.baseoperator import chain
import datetime
from airflow import models
from utils import xlml_controller

with models.DAG(
    dag_id='demo_dag',
    schedule='@daily',
    tags=['supported'],
    start_date=datetime.datetime(2023, 5, 18),
) as dag:
  jax_setup_cmds = (
      'pip install -U pip',
      'pip install --upgrade clu tensorflow tensorflow-datasets',
      'pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html',
      'git clone https://github.com/google/flax.git ~/flax',
      'pip install --user flax',
  )

  jax_resnet_cmds = (
      'cd ~/flax/examples/mnist',
      'JAX_PLATFORM_NAME=TPU python3 main.py --config=configs/default.py --workdir=/tmp/mnist --config.learning_rate=0.05 --config.num_epochs=3'
  )

  jax_resnet_test_v4_8 = xlml_controller.run(
      task_id_suffix='jax_resnet_test_v4_8',
      project='tpu-prod-env-one-vm',
      zone='us-central2-b',
      accelerator_type='v4-8',
      runtime_version='tpu-vm-v4-base',
      output_dir='/tmp/mnist',
      metric_name='test_accuracy',
      lower_bound=0.92,
      setup_cmds=jax_setup_cmds,
      run_cmds=jax_resnet_cmds,
  )

  jax_resnet_test_v4_32 = xlml_controller.run(
      task_id_suffix='jax_resnet_test_v4_32',
      project='tpu-prod-env-one-vm',
      zone='us-central2-b',
      accelerator_type='v4-8',
      runtime_version='tpu-vm-v4-base',
      output_dir='/tmp/mnist',
      metric_name='test_accuracy',
      lower_bound=0.92,
      setup_cmds=jax_setup_cmds,
      run_cmds=jax_resnet_cmds,
  )

  chain(jax_resnet_test_v4_8 >> jax_resnet_test_v4_32)