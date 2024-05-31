import os
import subprocess

def run_shell_script(script_path):
    process = subprocess.Popen(['bash', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b"" and process.poll() is not None:
            break
        if output:
            print(output.strip().decode())
    rc = process.poll()
    return rc

def main():
    script_path = os.path.join(os.path.dirname(__file__), 'deploy.sh')
    if os.path.exists(script_path):
        print(f"Ejecutando el script de despliegue: {script_path}")
        return_code = run_shell_script(script_path)

if __name__ == "__main__":
    main()
