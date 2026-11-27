import pytest

from spork_core.stable_refresh_exit import (
    ExitConfig, is_reversible_tool, refresh_allowed,
)


def call(name):
    return {"name": name, "arguments": {}}


READ_TOOLS = [
    "get_user_details", "get_flight_status", "get_reservation_details",
    "search_direct_flight", "search_onestop_flight",
    "list_all_airports", "list_all_product_types",
    "find_user_id_by_name_zip", "find_user_id_by_email",
]
WRITE_TOOLS = [
    "book_reservation", "cancel_reservation", "cancel_pending_order",
    "send_certificate", "update_reservation_flights",
    "update_reservation_baggages", "update_reservation_passengers",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "exchange_delivered_order_items", "return_delivered_order_items",
    "transfer_to_human_agents",
]


@pytest.mark.parametrize("name", READ_TOOLS)
def test_read_tools_are_reversible(name):
    assert is_reversible_tool(call(name))


@pytest.mark.parametrize("name", WRITE_TOOLS)
def test_write_tools_are_not_reversible(name):
    assert not is_reversible_tool(call(name))


def test_unknown_and_missing_calls_fail_closed():
    assert not is_reversible_tool(None)
    assert not is_reversible_tool({})
    assert not is_reversible_tool({"name": ""})


def test_read_only_policy_allows_only_read_to_read():
    read_a, read_b = call("get_user_details"), call("get_flight_status")
    write = call("send_certificate")
    assert refresh_allowed(read_a, read_b, "read_only")
    assert not refresh_allowed(read_a, write, "read_only")
    assert not refresh_allowed(write, read_a, "read_only")
    assert not refresh_allowed(write, write, "read_only")


def test_block_read_write_policy():
    read_a, read_b = call("get_user_details"), call("get_flight_status")
    write = call("send_certificate")
    assert refresh_allowed(read_a, read_b, "block_read_write")
    assert not refresh_allowed(read_a, write, "block_read_write")
    assert refresh_allowed(write, read_a, "block_read_write")
    assert refresh_allowed(write, write, "block_read_write")


def test_all_policy_preserves_old_behavior():
    assert refresh_allowed(call("get_user_details"), call("send_certificate"), "all")


def test_exit_config_validation():
    ExitConfig(refresh_policy="read_only").validate()
    ExitConfig(refresh_policy="block_read_write").validate()
    ExitConfig(refresh_policy="all").validate()
    with pytest.raises(ValueError):
        ExitConfig(refresh_policy="bogus").validate()
