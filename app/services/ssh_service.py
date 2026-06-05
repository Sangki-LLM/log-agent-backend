import base64
import io

import paramiko

from app.models.server import Server
from app.services.security_service import decrypt_pem


def _make_client(server: Server) -> paramiko.SSHClient:
    pem_text = decrypt_pem(server.pem_key_encrypted)
    key = paramiko.RSAKey.from_private_key(io.StringIO(pem_text))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=server.host, username=server.username, pkey=key, timeout=10)
    return client


def fetch_context_logs(server: Server, trigger_line: str, context_lines: int = 50) -> str:
    client = _make_client(server)
    try:
        cmd = (
            f'grep -n -E "error|exception|traceback" -i {server.log_path} | tail -n 5 | '
            f'awk -F: \'{{print $1}}\' | xargs -I{{}} sh -c '
            f'"sed -n \'$(({{}} - 5)),$(({{}} + {context_lines}))p\' {server.log_path}" 2>/dev/null || '
            f"tail -n {context_lines} {server.log_path}"
        )
        _, stdout, _ = client.exec_command(cmd)
        return stdout.read().decode(errors="replace")
    finally:
        client.close()


def fetch_source_file(server: Server, remote_path: str) -> str:
    client = _make_client(server)
    try:
        sftp = client.open_sftp()
        with sftp.open(remote_path, "r") as f:
            return f.read().decode(errors="replace")
    finally:
        client.close()


def write_source_file(server: Server, remote_path: str, content: str) -> None:
    client = _make_client(server)
    try:
        sftp = client.open_sftp()
        with sftp.open(remote_path, "w") as f:
            f.write(content)
    finally:
        client.close()


def run_command(server: Server, command: str) -> tuple[str, str]:
    client = _make_client(server)
    try:
        _, stdout, stderr = client.exec_command(command)
        return stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")
    finally:
        client.close()
