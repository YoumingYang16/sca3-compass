import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

from sca3_compass import molecular_data


def test_http_encoded_size_is_not_decoded_file_size(tmp_path, monkeypatch):
    content=b"File\tGSM123_example.tar.gz\tdate\t123\tTAR\n"*200
    encoded=gzip.compress(content)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Encoding","gzip")
            self.send_header("Content-Length",str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self,*args):
            pass
    server=HTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    real_client=httpx.Client
    class RedirectClient(real_client):
        def stream(self,method,url,**kwargs):
            return super().stream(method,f"http://127.0.0.1:{server.server_port}",**kwargs)
    monkeypatch.setattr(httpx,"Client",RedirectClient)
    monkeypatch.setattr(molecular_data,"PROJECT_ROOT",tmp_path)
    try:
        target=tmp_path/"filelist.txt"
        result=molecular_data.acquire("https://ftp.ncbi.nlm.nih.gov/test/filelist.txt",target)
        assert target.read_bytes()==content
        assert result["bytes"]==len(content)
        assert result["http_transfer_bytes"]==len(encoded)
        assert result["http_content_encoding"]=="gzip"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
