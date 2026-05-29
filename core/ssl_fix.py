import ssl
import os
import httpx

# ─────────────────────────────────────────────
# SSL Fix for restricted/corporate networks
# ─────────────────────────────────────────────
# langchain_mistralai creates its own internal httpx.Client, so global ssl context
# patches don't reach it. Instead, we monkey-patch httpx.Client.__init__ directly
# so ANY client created anywhere (including inside libraries) has verify=False.

os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

_original_httpx_init = httpx.Client.__init__

def _patched_httpx_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _original_httpx_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_httpx_init
