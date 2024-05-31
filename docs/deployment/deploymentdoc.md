# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** Modelo de Clasificación de Toxicidad
- **Plataforma de despliegue:** Servidor local o máquina virtual con Ubuntu 18.04 o superior
- **Requisitos técnicos:** 
    - Python 3.8+
    - Sistema operativo: Ubuntu 18.04 o superior
    - Git
    - Pip

- **Requisitos de seguridad:** 
- Autenticación de usuario para acceso al servidor.
- Encriptación de datos en tránsito utilizando HTTPS.
- Actualización regular de paquetes y dependencias.

- **Diagrama de arquitectura:** (imagen que muestra la arquitectura del sistema que se utilizará para desplegar el modelo)

## Código de despliegue

- **Archivo principal:** deploy.sh
- **Rutas de acceso a los archivos:** "docs/deployment/
- **Variables de entorno:** (lista de variables de entorno necesarias para el despliegue)
- 
## Documentación del despliegue

- **Instrucciones de instalación:** 
1. Clonar el Repositorio: 
git clone https://github.com/madboffin/U3_unal_proyecto.git
cd U3_unal_proyecto

2. Crear y Activar un Entorno Virtual:
python3 -m venv env
source env/bin/activate

3. Ejecutar script de despliegue:
python docs/deployment/deploy.py

- **Instrucciones de configuración:** (instrucciones detalladas para configurar el modelo en la plataforma de despliegue)
- **Instrucciones de uso:** python manage.py runserver
- **Instrucciones de mantenimiento:** 
1. Actualizar el Código:
        git pull origin main

2. Actualizar Dependencias:
        pip install -r requirements.txt
    
3. Realizar Migraciones de Base de Datos
        python manage.py migrate

4. Reiniciar el Servidor
        sudo systemctl restart nginx
        sudo systemctl restart gunicorn

