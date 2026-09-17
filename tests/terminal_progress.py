import fcntl
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time


binary = str(Path(sys.argv[1]).resolve())
record = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.5\n"


def run(interrupt):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.txt"
        source.write_bytes(record * (1_000_000 if interrupt else 1))

        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))

        env = {**os.environ, "TERM": "xterm-256color"}
        env.pop("CI", None)

        process = subprocess.Popen(
            [binary, "convert", "--from", "text", "--to", "bullet", "--input", str(source),
             "--output", str(Path(directory) / "output.bullet")],
            cwd=directory, stdin=slave, stdout=slave, stderr=slave, env=env, start_new_session=True,
        )

        output = bytearray()
        start = time.monotonic()
        resized = stopped = False

        try:
            while process.poll() is None:
                now = time.monotonic() - start
                if now > 15:
                    raise AssertionError("CLI did not exit promptly")

                if select.select([master], [], [], 0.02)[0]:
                    output.extend(os.read(master, 65536))

                if interrupt and b"Converting dataset" in output and not resized:
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 12, 35, 0, 0))
                    os.killpg(process.pid, signal.SIGWINCH)
                    resized = True

                elif resized and not stopped:
                    os.killpg(process.pid, signal.SIGINT)
                    stopped = True

            while select.select([master], [], [], 0.05)[0]:
                output.extend(os.read(master, 65536))

            assert termios.tcgetattr(slave) == original, "terminal settings changed"

            if interrupt:
                assert resized and stopped, output.decode(errors="replace")
                assert process.returncode != 0
                assert b"Interrupted" in output or b"Stopped before completion" in output
                assert not (Path(directory) / "output.bullet").exists()

            else:
                assert process.returncode == 0, output.decode(errors="replace")
                assert b"Complete." in output and b"1 pos" in output
                assert (Path(directory) / "output.bullet").stat().st_size == 32

        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

            os.close(master)
            os.close(slave)


run(False)
run(True)

print("PTY completion, resize, interruption and terminal restoration passed")
