import os
import glob
import subprocess
from tqdm import tqdm

#yaml_files = glob.glob("config/*.yaml") #("config/641-641-31-FR.yaml") # ("config/*.yaml")
yaml_files = ['config/549-607-0-FR.yaml', 'config/549-607-0-MD.yaml', 'config/549-607-0-RR.yaml',]
for yaml_file in tqdm(yaml_files):
    config_name = os.path.splitext(os.path.basename(yaml_file))[0]
    output_file = f"trace_{config_name}.txt"
    print(f"Running with config: {config_name}")

    with open(output_file, 'w') as f:
        process = subprocess.Popen(["python", "sample_multiagent_check.py", f"--config-name={config_name}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            print(line.strip())
            f.write(line)
        
        process.wait()

