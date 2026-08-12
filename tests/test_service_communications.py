from microgrid_simulator.ui.server import _message_route, _service_communications_payload


def test_message_route_names_project_requests_and_replies() -> None:
    assert _message_route("mgs.v1.plant.step") == {
        "kind": "request",
        "source": "EMS",
        "target": "Headless plant",
    }
    assert _message_route("mgs.v1.dashboard.session.row")["kind"] == "event"
    assert _message_route("_INBOX.example.reply")["kind"] == "reply"


def test_service_communications_payload_turns_connz_into_readable_paths() -> None:
    payload = _service_communications_payload(
        {
            "version": "2.11.0",
            "connections": 4,
            "subscriptions": 3,
            "in_msgs": 20,
            "out_msgs": 18,
            "in_msgs_per_sec": 2.5,
            "out_msgs_per_sec": 2.0,
            "jetstream": {"config": {"store_dir": "/data"}},
        },
        {
            "connections": [
                {
                    "name": "microgrid-telemetry",
                    "in_msgs": 2,
                    "out_msgs": 2,
                    "subscriptions_list_detail": [
                        {
                            "subject": "mgs.v1.telemetry.window",
                            "qgroup": "telemetry-workers",
                            "msgs": 2,
                        }
                    ],
                },
                {
                    "name": "microgrid-ems",
                    "in_msgs": 8,
                    "out_msgs": 8,
                    "subscriptions_list_detail": [],
                },
                {
                    "name": "microgrid-plant",
                    "in_msgs": 10,
                    "out_msgs": 8,
                    "subscriptions_list_detail": [
                        {"subject": "mgs.v1.plant.step", "qgroup": "plant-workers", "msgs": 4},
                        {"subject": "mgs.v1.plant.start", "qgroup": "plant-workers", "msgs": 1},
                    ],
                },
            ]
        },
    )

    assert payload["status"] == "ok"
    assert payload["server"]["jetstream"] is True

    services = {service["id"]: service for service in payload["services"]}
    assert services["microgrid-ems"]["status"] == "connected"
    assert services["microgrid-plant"]["subscriptions"][0]["queue"] == "plant-workers"
    assert services["microgrid-telemetry"]["subscriptions"][0]["messages"] == 2

    paths = {path["id"]: path for path in payload["paths"]}
    assert paths["ems-telemetry"]["status"] == "active"
    assert paths["ems-telemetry"]["delivered_messages"] == 2
    assert paths["ems-plant-step"]["delivered_messages"] == 4
    assert paths["ems-control-room"]["status"] == "available"


def test_service_communications_payload_degrades_without_nats_monitoring() -> None:
    payload = _service_communications_payload({}, {}, monitor_error="connection refused")

    assert payload["status"] == "degraded"
    assert payload["error"] == "connection refused"
    services = {service["id"]: service for service in payload["services"]}
    assert services["nats"]["status"] == "unavailable"
    assert services["microgrid-ems"]["status"] == "not connected"
    paths = {path["id"]: path for path in payload["paths"]}
    assert paths["ems-plant-step"]["status"] == "unknown"
    assert paths["ems-control-room"]["status"] == "available"
