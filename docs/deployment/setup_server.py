import os
import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.strip().decode())

def install_dependencies():
    run_command("sudo apt-get update")
    run_command("sudo apt-get install -y python3 python3-pip python3-venv nginx git")

def configure_nginx(project_dir):
    nginx_config = f"""
    server {{
        listen 80;
        server_name localhost;

        location / {{
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}

        location /static/ {{
            alias {project_dir}/static;
        }}
    }}
    """
    config_path = "/etc/nginx/sites-available/project"
    with open(config_path, "w") as f:
        f.write(nginx_config)
    run_command(f"sudo ln -s {config_path} /etc/nginx/sites-enabled")
    run_command("sudo nginx -t")
    run_command("sudo systemctl restart nginx")

def main():
    repo_url = "https://github.com/madboffin/U3_unal_proyecto.git"
    project_dir = "U3_unal_proyecto"

    install_dependencies()
    run_command(f"git clone {repo_url} {project_dir}")
    os.chdir(project_dir)
    configure_nginx(os.getcwd())
    run_command("bash docs/deployment/deploy.sh")
    run_command("python3 docs/deployment/deploy.py")

if __name__ == "__main__":
    main()
