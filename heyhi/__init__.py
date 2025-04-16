# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple
import copy
import datetime
import enum
import hashlib
import functools
import logging
import os
import pathlib
import shutil
import subprocess
import time

import hydra
import submitit
import torch


ModeType = str

MAX_EXP_LEN = 150


MODES: Tuple[ModeType, ...] = (
    "gentle_start",
    "start_restart",
    "start_continue",
    "restart",
    "dryrun",
    "kill",
)
JOBFILE_NAME = "heyhi.jobid"
RESULTFILE_NAME = "result.torch"
DELIMETER = "@"
LOCAL_JOB_ID = "local"
_SLURM_CACHE = {}


def is_aws():
    return os.uname()[1].startswith("ip-")


def setup_logging():
    """Enable pretty logging and sets the level to DEBUG."""
    logging.addLevelName(logging.DEBUG, "D")
    logging.addLevelName(logging.INFO, "I")
    logging.addLevelName(logging.WARNING, "W")
    logging.addLevelName(logging.ERROR, "E")
    logging.addLevelName(logging.CRITICAL, "C")

    formatter = logging.Formatter(
        fmt=("%(levelname)s%(asctime)s" " [%(module)s:%(lineno)d] %(message)s"),
        datefmt="%m%d %H:%M:%S",
    )

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(console_handler)

    return logger


def log_git_status():
    git_repo = str(pathlib.Path(__file__).resolve().parent.parent)
    try:
        rev = subprocess.check_output("git rev-parse HEAD".split(), cwd=git_repo)
    except subprocess.CalledProcessError as e:
        logging.error("Attempt to call 'git rev-parse HEAD' failed: %s", e)
    else:
        logging.info("Git revision: %s", rev.decode("utf8").strip())
    try:
        diff = subprocess.check_output("git diff HEAD".split(), cwd=git_repo)
    except subprocess.CalledProcessError as e:
        logging.error("Attempt to call 'git diff HEAD", e)
    else:
        if diff:
            diff_path = pathlib.Path("workdir.diff").resolve()
            if is_master():
                logging.info("Found unsubmitited diff. Saving to %s", diff_path)
                with diff_path.open("w") as stream:
                    stream.write(diff.decode("utf8"))
            else:
                logging.info("Found unsubmitited diff. NOT saving as not the master.")
        else:
            logging.info("No diff in the working copy")


# def _get_all_runing_job_ids(user_only: bool = False) -> FrozenSet[str]:
#     cmd = ["squeue", "-r", "-h", "-o", "%i"]
#     if user_only:
#         cmd.extend(["-u", os.environ["USER"]])
#     output = subprocess.check_output(cmd)
#     job_ids = output.decode("utf8").split()
#     return frozenset(job_ids)
def _get_all_runing_job_ids(user_only: bool = False) -> FrozenSet[str]:
    # 完全删除原有逻辑，直接返回空集合
    return frozenset()

def get_all_runing_job_ids() -> FrozenSet[str]:
    global _SLURM_CACHE
    if "job_list" not in _SLURM_CACHE:
        _SLURM_CACHE["job_list"] = _get_all_runing_job_ids()
    return _SLURM_CACHE["job_list"]


class Status(enum.IntEnum):
    NOT_STARTED = 1
    DONE = 2
    DEAD = 3
    RUNNING = 4


def is_on_slurm() -> bool:
    return "SLURM_PROCID" in os.environ


def get_slurm_job_id() -> Optional[str]:
    return os.environ.get("SLURM_JOBID")


def is_master() -> bool:
    return os.environ.get("SLURM_PROCID", "0") == "0"


class ExperimentDir:
    def __init__(self, root_exp_dir: pathlib.Path, exp_id: str):
        self.exp_path = root_exp_dir / exp_id
        self.exp_id = exp_id

    @property
    def job_id_path(self) -> pathlib.Path:
        return self.exp_path / JOBFILE_NAME

    @property
    def result_path(self) -> pathlib.Path:
        return self.exp_path / RESULTFILE_NAME

    def maybe_get_job_id(self) -> Optional[str]:
        if self.job_id_path.exists():
            with self.job_id_path.open() as stream:
                return stream.read().strip()
        return None

    def save_job_id(self, job_id: str) -> None:
        self.job_id_path.parent.mkdir(exist_ok=True, parents=True)
        with self.job_id_path.open("w") as stream:
            print(job_id, file=stream)

    def get_status(self) -> Status:
        if not self.job_id_path.exists():
            if self.exp_path.exists():
                logging.warning(
                    "Experiment folder without job_id file: %s", self.exp_path
                )
            return Status.NOT_STARTED
        maybe_jobid = self.maybe_get_job_id()
        if maybe_jobid in get_all_runing_job_ids():
            return Status.RUNNING
        if self.result_path.exists():
            return Status.DONE
        return Status.DEAD

    def is_done(self) -> bool:
        return self.get_status() == Status.DONE

    def is_started(self) -> bool:
        return self.get_status() != Status.NOT_STARTED

    def is_running(self) -> bool:
        return self.get_status() == Status.RUNNING

    def kill(self, silent: bool) -> None:
        if not self.exp_path.exists():
            return
        logging.info("Kill for %s", self.exp_path)
        maybe_jobid = self.maybe_get_job_id()
        if maybe_jobid is not None and maybe_jobid != LOCAL_JOB_ID:
            if not silent:
                print("killing job", maybe_jobid, "...", end="", flush=True)
            subprocess.check_call(["scancel", str(maybe_jobid)])
        if not silent:
            print("done")

    def kill_and_prune(self, silent: bool) -> None:
        if not self.exp_path.exists():
            return
        logging.info("Prune+kill for %s", self.exp_path)
        if not silent:
            print("Deleting the folder in 3 seconds", end="", flush=True)
            for _ in range(3):
                print(".", end="", flush=True)
                time.sleep(1)
        maybe_jobid = self.maybe_get_job_id()
        if maybe_jobid is not None and maybe_jobid != LOCAL_JOB_ID:
            if not silent:
                print("killing job", maybe_jobid, "...", end="", flush=True)
            subprocess.check_call(["scancel", str(maybe_jobid)])
        if not silent:
            print(" purging the log dir", "...", end="", flush=True)
        shutil.rmtree(str(self.exp_path))
        if not silent:
            print("done")

    @property
    def slurm_path(self) -> pathlib.Path:
        return self.exp_path / "slurm"


def save_result_in_cwd(f):
    """Save results of the function to a results.torch file in cwd."""

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        result = f(*args, **kwargs)
        result_path = os.path.join(os.getcwd(), RESULTFILE_NAME)
        if is_master():
            logging.info("Saving result to %s", result_path)
            torch.save(result, result_path)
        return result

    return wrapped


def _sort_overrides(overrides: Sequence[str]) -> List[str]:
    def sort_key(override):
        name = override.split("=")[0]
        depth = len(name.split("."))
        return (depth, name, override)

    return sorted(overrides, key=sort_key)


def _parse_overrides_quick(overrides: Sequence[str]) -> Dict[str, str]:
    d = {}
    for override in overrides:
        try:
            name, value = override.split("=", 1)
        except ValueError:
            raise ValueError(f"Bad override: {override}")
        d[name] = value
    return d


def handle_dst(
    root_exp_dir: pathlib.Path,
    mode: ModeType,
    config_path: pathlib.Path,
    overrides: Sequence[str],
    adhoc: bool,
    force_override_exp_id: Optional[str] = None,
    force_override_tag: Optional[str] = None,
) -> Tuple[ExperimentDir, bool]:
    """创建/重新创建 ExperimentDir 并检查是否需要执行操作。

    该函数的逻辑如下：
        1. 从 config_path 和 overrides 创建基础 exp_id。
        1a. 如果设置了 force_override_tag，它将替代自动生成的 overrides 字符串。
        1b. 如果提供了 force_override_exp_id，它将替代配置路径部分。
        2. 如果在 adhoc 模式下运行，在 exp_id 前添加 'adhoc/<date>' 以获取唯一的 exp_id。
           否则，添加 'p/'（永久）。
        3. 获取文件夹中作业的状态：NOT_STARTED（未开始）、RUNNING（运行中）、DONE（完成）、
           DEAD（终止）。
        4. 根据模式可能需要清除文件夹并终止作业，以及设置 need_run 标志。

    如果模式是：
        - gentle_start：仅当作业为 NOT_STARTED 时设置 need_run。
        - start_restart：当作业为 NOT_STARTED 或 DEAD 时设置 need_run。
          如果作业已终止，清除实验目录。
        - start_continue：当作业为 NOT_STARTED 或 DEAD 时设置 need_run。
        - restart：将 need_run 设置为 True，如果作业正在运行则终止它。
        - dryrun：将 need_run 设置为 False。

    返回一个元组 (ExperimentDir, need_run)。
    """
    assert config_path.exists(), config_path
    *_, folder_name, config_name = str(config_path.absolute()).split("/")
    config_name = config_name.rsplit(".", 1)[0]
    assert DELIMETER not in folder_name, folder_name
    assert DELIMETER not in config_name, config_name
    base_exp_id = f"{folder_name}/{config_name}"
    if force_override_exp_id is not None:
        logging.warning("Exp id override: %s -> %s", base_exp_id, force_override_exp_id)
        base_exp_id = force_override_exp_id
    if force_override_tag is not None:
        exp_id = DELIMETER.join([base_exp_id, force_override_tag])
    elif overrides:
        exp_id = DELIMETER.join([base_exp_id] + _sort_overrides(overrides))
        exp_id = exp_id.replace("=", DELIMETER).replace(" ", "_")
    else:
        exp_id = base_exp_id

    if len(exp_id) > MAX_EXP_LEN:
        logging.warning("Name of experiment is too long: %s", exp_id)
        exp_id = "%s_%s" % (
            exp_id[:50],
            hashlib.md5(exp_id.encode("utf8")).hexdigest()[:16],
        )
        logging.warning("Shortened name: %s", exp_id)

    if adhoc:
        date = datetime.datetime.now().isoformat()
        exp_id = "adhoc/%s/%s" % (date, exp_id)
    else:
        # Permanent.
        exp_id = "p/%s" % exp_id

    need_run = True
    exp_handle = ExperimentDir(root_exp_dir, exp_id)
    if mode == "gentle_start":
        if exp_handle.is_started():
            logging.info("Alredy started. Status: %s", exp_handle.get_status())
            need_run = False

    elif mode == "start_restart":
        if exp_handle.is_running() or exp_handle.is_done():
            logging.info("Running or done. Status: %s", exp_handle.get_status())
            need_run = False
        elif not exp_handle.is_done():
            exp_handle.kill_and_prune(silent=False)
    elif mode == "start_continue":
        if exp_handle.is_running() or exp_handle.is_done():
            logging.info("Running or done. Status: %s", exp_handle.get_status())
            need_run = False
    elif mode == "restart":
        exp_handle.kill_and_prune(silent=False)
    elif mode == "kill":
        exp_handle.kill(silent=False)
    elif mode == "dryrun":
        logging.info("Dry run, not starting anything")
        need_run = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return exp_handle, need_run


def _build_slurm_executor(exp_handle, cfg):
    executor = submitit.SlurmExecutor(folder=exp_handle.slurm_path)
    assert cfg.num_gpus < 8 or cfg.num_gpus % 8 == 0, cfg.num_gpus
    if cfg.num_gpus:
        gpus = min(cfg.num_gpus, 8)
        nodes = max(1, cfg.num_gpus // 8)
        assert (
            gpus * nodes == cfg.num_gpus
        ), "Must use 8 gpus per machine when multiple nodes are used."
    else:
        gpus = 0
        nodes = 1

    if cfg.single_task_per_node:
        ntasks_per_node = 1
    else:
        ntasks_per_node = gpus

    slurm_params = dict(
        job_name=exp_handle.exp_id,
        partition=cfg.partition,
        time=int(cfg.hours * 60),
        nodes=nodes,
        num_gpus=gpus,
        ntasks_per_node=ntasks_per_node,
        mem=f"{cfg.mem_per_gpu * max(1, gpus)}GB",
        signal_delay_s=90,
        comment=cfg.comment or "",
    )
    if cfg.cpus_per_gpu:
        slurm_params["cpus_per_task"] = cfg.cpus_per_gpu * gpus // ntasks_per_node
    if cfg.volta32:
        slurm_params["constraint"] = "volta32gb"
    if cfg.pascal:
        slurm_params["constraint"] = "pascal"
    if cfg.volta:
        slurm_params["constraint"] = "volta"
    if is_aws():
        slurm_params["mem"] = 0
        slurm_params["cpus_per_task"] = 1
        slurm_params["partition"] = "compute"
        if "constraint" in slurm_params:
            del slurm_params["constraint"]
    logging.info("Slurm params: %s", slurm_params)
    executor.update_parameters(**slurm_params)
    return executor


def run_with_config(
    task_function: Callable,  # 任务函数
    exp_handle: ExperimentDir,  # 实验目录句柄
    config_path: pathlib.Path,  # 配置文件路径
    overrides: Sequence[str],  # 覆盖参数列表
    config_search_paths: Optional[Sequence[pathlib.Path]] = None,  # 可选的配置搜索路径
) -> None:
    setup_logging()  # 设置日志
    del config_search_paths  # 暂时硬编码在hydra中
    # 构建hydra参数列表
    hydra_args = list(overrides) + [
        f"hydra.run.dir={exp_handle.exp_path}",  # 设置运行目录
        f"job_id={exp_handle.exp_id}",  # 设置作业ID
        # 移除hydra默认配置
        "hydra.hydra_logging=null",
        "hydra.job_logging=null", 
        "hydra.sweep=null",
        "hydra.sweeper=null",
    ]
    logging.info("Passing the following args to hydra: %s", hydra_args)  # 记录传递给hydra的参数
    calling_file = "train.py"  # 调用文件名
    abs_base_dir = os.path.realpath(os.path.dirname(calling_file))  # 获取绝对基础目录
    hydra_config_path = str(config_path.resolve())  # 解析配置文件路径
    # 验证配置文件路径是否在基础目录下
    assert hydra_config_path.startswith(abs_base_dir + "/"), (
        hydra_config_path,
        abs_base_dir,
    )
    hydra_config_path = hydra_config_path[len(abs_base_dir) + 1 :]  # 获取相对路径
    # 创建hydra对象
    hydra_obj = hydra.Hydra(
        calling_file="run.py",  # 调用文件
        calling_module=None,  # 调用模块
        config_path=hydra_config_path,  # 配置路径
        task_function=task_function,  # 任务函数
        verbose="",  # 详细程度
        strict=False,  # 是否严格模式
    )
    cfg = hydra_obj._load_config(copy.deepcopy(hydra_args))  # 加载配置
    use_slurm = False  # 是否使用slurm
    if cfg.launcher is not None:  # 如果配置了启动器
        assert cfg.launcher.driver in ("slurm", "local"), cfg.launcher  # 验证启动器类型
        use_slurm = cfg.launcher.driver == "slurm"  # 设置是否使用slurm
    if use_slurm and not is_on_slurm():  # 如果需要使用slurm且不在slurm环境中
        logging.info("Config:\n%s", cfg.pretty())  # 记录配置信息
        executor = _build_slurm_executor(exp_handle, cfg.launcher)  # 构建slurm执行器
        job = executor.submit(hydra_obj.run, overrides=hydra_args)  # 提交作业
        exp_handle.save_job_id(job.job_id)  # 保存作业ID
        logging.info("Submitted job %s", job.job_id)  # 记录提交的作业ID
        # 记录日志文件路径
        logging.info(
            "stdout:\n\ttail -F %s/%s_0_log.out", exp_handle.slurm_path, job.job_id
        )
        logging.info(
            "stderr:\n\ttail -F %s/%s_0_log.err", exp_handle.slurm_path, job.job_id
        )
    else:  # 如果不需要使用slurm或已在slurm环境中
        exp_handle.save_job_id(LOCAL_JOB_ID)  # 保存本地作业ID
        hydra_obj.run(overrides=copy.deepcopy(hydra_args))  # 直接运行任务
