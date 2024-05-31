
# Actualizar paquetes
sudo apt-get update
sudo apt-get install -y python3 python3-pip git

git clone https://github.com/madboffin/U3_unal_proyecto.git
cd U3_unal_proyecto

# Activar entorno virtual
python3 -m venv env
source env/bin/activate

# Dependencias
pip install -r requirements.txt

python scripts/training/main.py