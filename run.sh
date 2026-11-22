#!/usr/bin/env sh
# Termux (teléfono) o PC. Para IA completa en PC: pip install -r requirements.txt
cd "$(dirname "$0")"
if [ -d /data/data/com.termux ]; then
  pkg install -y python ffmpeg python-numpy rust clang make libffi openssl
fi
pip install -r requirements-lite.txt
echo "Abre http://localhost:8000 en el navegador del teléfono (o http://IP-DEL-PC:8000 desde el teléfono)"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
