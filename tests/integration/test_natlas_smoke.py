import os

import pytest

from ncair_lms.config import Settings
from ncair_lms.models import ToolName
from ncair_lms.natlas import LocalNatlasClient, NatlasRouter


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("NCAIR_RUN_INTEGRATION") != "1",
    reason="set NCAIR_RUN_INTEGRATION=1 to load the real N-ATLaS model",
)
def test_real_natlas_routes_a_hausa_login_request():
    settings = Settings.from_env()
    router = NatlasRouter(LocalNatlasClient(settings))
    decision = router.route("Ina zan shiga LMS dina?")
    assert decision.tool is ToolName.PORTAL_LINK
