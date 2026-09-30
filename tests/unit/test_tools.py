import pytest

from ncair_lms.errors import InvalidToolArgumentsError
from ncair_lms.models import PortalAction
from ncair_lms.tools import get_portal_link, get_step_guidance


def test_portal_aliases_share_canonical_urls():
    assert get_portal_link(PortalAction.LOGIN).answer == get_portal_link(PortalAction.SIGNIN).answer
    assert "intern/courses" in get_portal_link(PortalAction.TRACK_SELECTION).answer


def test_step_guidance_accepts_only_known_steps():
    assert "PSIN 50-Seater Hall" in get_step_guidance(2).answer
    with pytest.raises(InvalidToolArgumentsError):
        get_step_guidance(5)
