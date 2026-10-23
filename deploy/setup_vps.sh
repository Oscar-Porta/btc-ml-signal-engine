#!/bin/bash
# ==========================================================
# SETUP VPS — Trading Signal Project
# ==========================================================
# Ejecutar DESPUÉS de haber copiado el proyecto al servidor.
# El script asume que estás dentro de la carpeta del proyecto.
#
# Uso:
#   cd /root/trading_project_signal   (o donde lo hayas copiado)
#   chmod +x deploy/setup_vps.sh
#   sudo ./deploy/setup_vps.sh
# ==========================================================

set -e  # Parar si hay error

# Detectar directorio del proyecto
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="/home/trading/trading_project_signal"

echo "=========================================="
echo "  CONFIGURACIÓN VPS - Trading Signal"
echo "=========================================="
echo "  Proyecto origen: $SOURCE_DIR"
echo ""

# --- 1. Actualizar sistema ---
echo "[1/8] Actualizando sistema..."
apt update && apt upgrade -y

# --- 2. Instalar Python y herramientas ---
echo "[2/8] Instalando Python 3 y dependencias del sistema..."
apt install -y python3 python3-pip python3-venv git curl

# --- 3. Crear usuario para el proyecto ---
echo "[3/8] Creando usuario 'trading'..."
if ! id "trading" &>/dev/null; then
    useradd -m -s /bin/bash trading
    echo "  Usuario 'trading' creado."
else
    echo "  Usuario 'trading' ya existe."
fi

# --- 4. Copiar proyecto a /home/trading ---
echo "[4/8] Copiando proyecto a $PROJECT_DIR..."
mkdir -p "$PROJECT_DIR"
cp -r "$SOURCE_DIR"/* "$PROJECT_DIR"/
cp -r "$SOURCE_DIR"/.env "$PROJECT_DIR"/ 2>/dev/null || true
chown -R trading:trading "$PROJECT_DIR"

# --- 5. Entorno virtual + dependencias ---
echo "[5/8] Creando entorno virtual e instalando librerías..."
cd "$PROJECT_DIR"

# Eliminar venv viejo si existe (puede ser de otra máquina)
rm -rf .venv

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "  Dependencias instaladas."

# --- 6. Copiar servicios systemd ---
echo "[6/8] Configurando servicios systemd..."
cp deploy/trading-signal.service /etc/systemd/system/
cp deploy/trading-dashboard.service /etc/systemd/system/
systemctl daemon-reload

# --- 7. Abrir puerto del dashboard ---
echo "[7/8] Configurando firewall..."
if command -v ufw &>/dev/null; then
    ufw allow ssh 2>/dev/null || true
    ufw allow 8501/tcp 2>/dev/null || true
    echo "  Puerto 8501 abierto."
else
    echo "  ufw no disponible. Abre el puerto 8501 manualmente."
fi

# --- 8. Habilitar servicios ---
echo "[8/8] Habilitando servicios (arrancan solos tras reboot)..."
systemctl enable trading-signal
systemctl enable trading-dashboard

# Ajustar permisos finales
chown -R trading:trading "$PROJECT_DIR"

echo ""
echo "=========================================="
echo "  CONFIGURACIÓN COMPLETADA"
echo "=========================================="
echo ""
echo "  ANTES de iniciar los servicios, asegúrate de que"
echo "  el archivo .env existe y tiene los tokens de Telegram:"
echo ""
echo "    nano $PROJECT_DIR/.env"
echo ""
echo "  Contenido del .env:"
echo "    TELEGRAM_BOT_TOKEN=tu_token"
echo "    TELEGRAM_CHAT_ID=tu_chat_id"
echo ""
echo "  Para iniciar todo:"
echo "    sudo systemctl start trading-signal"
echo "    sudo systemctl start trading-dashboard"
echo ""
echo "  Para ver logs en tiempo real:"
echo "    journalctl -u trading-signal -f"
echo "    journalctl -u trading-dashboard -f"
echo ""
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8501"
echo "=========================================="
