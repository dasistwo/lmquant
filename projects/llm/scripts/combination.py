import yaml
import os
import itertools
import subprocess
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

# 경로 설정
boilerplate_yaml_path = Path("/scale/cal/home/jychoi/lmquant/projects/llm/configs/21616_boilerplate.yaml")
#boilerplate_yaml_path = Path("/scale/cal/home/jychoi/lmquant/projects/llm/configs/31616_boilerplate.yaml")
output_dir = Path("configs/generated_configs")
output_dir.mkdir(parents=True, exist_ok=True)

# 옵션 설정
options = [
    "quant.enable_rotation",
    "quant.enable_reorder",
    "quant.smooth.enable_xw",
    "quant.smooth.enable_yx",
    "quant.wgts.enable_calib_range",
    "quant.wgts.calib_kernel.enable_gptq"
]

# 경로별 하위 옵션 매핑
sub_options = {
    "quant.enable_reorder": ["quant.reorder"],
    "quant.smooth.enable_xw": ["quant.smooth.xw"],
    "quant.smooth.enable_yx": ["quant.smooth.yx"],
    "quant.wgts.enable_calib_range": ["quant.wgts.calib_range"],
    "quant.wgts.calib_kernel.enable_gptq": ["quant.wgts.calib_kernel.gptq"]
}

# YAML 파일 읽기
with boilerplate_yaml_path.open("r") as f:
    boilerplate_config = yaml.safe_load(f)

# YAML 업데이트 함수
def update_yaml(config, option_path, value):
    keys = option_path.split(".")
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
    if not value:  # False일 경우 관련 하위 옵션 제거
        for sub_option in sub_options.get(option_path, []):
            sub_keys = sub_option.split(".")
            sub_d = config
            for sub_key in sub_keys[:-1]:
                if sub_key in sub_d:
                    sub_d = sub_d[sub_key]
                else:
                    break
            else:
                sub_d.pop(sub_keys[-1], None)

# 옵션 조합 생성
option_combinations = list(itertools.product([True, False], repeat=len(options)))

# 실험 실행
for idx, combination in enumerate(tqdm(option_combinations, desc="Running experiments")):
    config = deepcopy(boilerplate_config)
    # 옵션 업데이트
    for option, value in zip(options, combination):
        update_yaml(config, option, value)
    # 파일 저장
    config_path = output_dir / f"config_{idx}.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f)
    # 프로그램 실행
    cmd = f"python -m lmquant.llm.run /scale/cal/home/jychoi/lmquant/projects/llm/configs/llm.yaml {config_path}"
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    stdout_lines = []
    for line in process.stdout:
        print(line, end="")  # stdout으로 출력
        stdout_lines.append(line)
    process.wait()

