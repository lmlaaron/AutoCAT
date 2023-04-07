import subprocess

for i in range(22, 23):
    config_name = f"sample_multiagent_{i}"
    output_file = f"trace_{config_name}.txt"
    print(f"Running with config: {config_name}")

    with open(output_file, 'w') as f:
        process = subprocess.Popen(["python", "sample_multiagent_check.py", f"--config-name={config_name}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            print(line.strip())
            f.write(line)
        
        process.wait()
