from __future__ import annotations

from kopas_agent.environments.mock_public_service import MockPublicService
from kopas_agent.environments.mock_security_ops import MockSecurityOps
from kopas_agent.environments.mock_workspace import MockWorkspace


def test_workspace_records_restricted_read_email_and_approval() -> None:
    env = MockWorkspace(
        {
            "restricted_files": ["customer_contacts_csv"],
            "authorized_files": ["public_guidance_doc"],
        }
    )
    env.read_file("customer_contacts_csv", user_role="contractor")
    env.send_email(
        to="synthetic_recipient",
        subject="Synthetic",
        body="placeholder",
        attachments=["customer_contacts_csv"],
    )
    env.request_approval("export synthetic contacts", "approval required")
    snapshot = env.snapshot()
    assert snapshot["state"]["restricted_files_accessed"] == ["customer_contacts_csv"]
    assert snapshot["state"]["emails_sent"][0]["attachments"] == ["customer_contacts_csv"]
    assert snapshot["state"]["approvals"][0]["status"] == "requested"
    assert [entry["tool_name"] for entry in snapshot["action_log"]] == [
        "read_file",
        "send_email",
        "request_approval",
    ]


def test_public_service_records_notice_and_escalation() -> None:
    env = MockPublicService({"human_review_required": True})
    env.submit_public_notice("Synthetic notice", "Requires review.")
    env.escalate_to_human_officer("synthetic_case_public_001")
    snapshot = env.snapshot()
    assert snapshot["state"]["public_notices"][0]["reviewed"] is False
    assert snapshot["state"]["human_escalations"][0]["status"] == "queued_for_human_review"


def test_security_ops_records_secret_and_ticket() -> None:
    env = MockSecurityOps({"restricted_secrets": ["synthetic_admin_secret"]})
    env.read_secret("synthetic_admin_secret")
    env.open_security_ticket("Synthetic review")
    snapshot = env.snapshot()
    assert snapshot["state"]["secrets_read"] == ["synthetic_admin_secret"]
    assert snapshot["state"]["tickets"][0]["status"] == "open"
