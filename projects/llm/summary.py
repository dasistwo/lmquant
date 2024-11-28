import os
import yaml
import json
import csv

# 최상위 폴더 경로 설정
top_folder = "/scale/cal/home/jychoi/model/qserve_checkpoints/llm/llama2/llama2-7b/w.3-x.16-y.16/w.zint3-x.fp16-y.fp16"

# 추출할 옵션 목록 및 경로
options = {
    "quant.enable_rotation": ["quant", "enable_rotation"],
    "quant.enable_reorder": ["quant", "enable_reorder"],
    "quant.enable_smooth": ["quant", "enable_smooth"],
    "quant.smooth.enable_xw": ["quant", "smooth", "enable_xw"],
    "quant.smooth.enable_yx": ["quant", "smooth", "enable_yx"],
    "quant.wgts.enable_calib_range": ["quant", "wgts", "enable_calib_range"],
}

# 결과를 저장할 리스트
results = []

# 하위 폴더 순회
for root, dirs, files in os.walk(top_folder):
    if "config.yaml" in files and "results.json" in files:
        config_path = os.path.join(root, "config.yaml")
        result_path = os.path.join(root, "results.json")

        # YAML 파일 읽기
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # JSON 파일 읽기
        with open(result_path, "r") as f:
            result_data = json.load(f)

        # 옵션 값 추출
        config_values = {}
        for opt, path in options.items():
            value = config_data
            for key in path:
                value = value.get(key, None)
                if value is None:
                    break
            config_values[opt] = value

    
        # 조건에 따라 smooth 관련 값 처리
        if config_values.get("quant.enable_smooth", False) is False:
            # quant.enable_smooth이 False라면 둘 다 False
            config_values["quant.smooth.enable_xw"] = False
            config_values["quant.smooth.enable_yx"] = False
        
        # 결과 데이터 추출 (예: word_perplexity)
        word_perplexity = result_data.get("2048", {}).get("results", {}).get("wikitext", {}).get("word_perplexity", None)

        # 데이터 추가
        results.append({**config_values, "word_perplexity": word_perplexity, "folder": root})

# CSV 파일로 저장
output_file = "results_summary.csv"
with open(output_file, "w", newline="") as csvfile:
    fieldnames = list(options.keys()) + ["word_perplexity", "folder"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_file}")

