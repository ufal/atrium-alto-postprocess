import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from service.text_api import app

client = TestClient(app)


def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "quality_categories" in data
    assert "Clear" in data["quality_categories"]


@patch("service.text_api.text_manager.process_text_file", create=True)
def test_process_text_auto_routing(mock_process):
    """Ensure text uploads correctly route to the text_manager text processor."""
    mock_process.return_value = {
        "type": "plain_text",
        "cleaned_lines": [{"line_num": 1, "text": "Mocked Line", "category": "Clear"}]
    }

    content = b"Mock line content"
    files = {"file": ("document.txt", content, "text/plain")}
    data = {"task_type": "auto"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["type"] == "plain_text"
    assert res_data["cleaned_lines"][0]["category"] == "Clear"
    assert res_data["filename"] == "document.txt"


@patch("service.text_api.text_manager.process_alto", create=True)
def test_process_alto_explicit_routing(mock_process):
    """Ensure ALTO XML uploads hit the alto pipeline specifically."""
    mock_process.return_value = {"type": "alto_xml", "cleaned_lines": []}

    content = b"<alto></alto>"
    files = {"file": ("document.xml", content, "application/xml")}
    data = {"task_type": "alto"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["type"] == "alto_xml"