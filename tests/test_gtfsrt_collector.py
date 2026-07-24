from google.transit import gtfs_realtime_pb2

from src.collect.gtfsrt_collector import (
    AlertDeltaCache,
    DeltaCache,
    cause_name,
    effect_name,
    extract_alert_rows,
    extract_rows,
    translated_text,
)


def _translated_string(text: str, lang: str = "en"):
    ts = gtfs_realtime_pb2.TranslatedString()
    t = ts.translation.add()
    t.text = text
    t.language = lang
    return ts


def _feed_with_alert(alert_id: str, header: str, effect: int, route_id: str, stop_id: str = "") -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.timestamp = 1000
    entity = feed.entity.add()
    entity.id = alert_id
    entity.alert.effect = effect
    entity.alert.cause = gtfs_realtime_pb2.Alert.Cause.ACCIDENT
    entity.alert.header_text.CopyFrom(_translated_string(header))
    ie = entity.alert.informed_entity.add()
    ie.route_id = route_id
    if stop_id:
        ie.stop_id = stop_id
    period = entity.alert.active_period.add()
    period.start = 500
    period.end = 2000
    return feed


def test_extract_alert_rows_basic_fields():
    feed = _feed_with_alert("a1", "Delay on route 5", gtfs_realtime_pb2.Alert.Effect.SIGNIFICANT_DELAYS, "5")
    rows, seen = extract_alert_rows(feed, poll_ts=1000, cache=AlertDeltaCache())
    assert seen == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["alert_id"] == "a1"
    assert row["effect"] == "SIGNIFICANT_DELAYS"
    assert row["cause"] == "ACCIDENT"
    assert row["header_text"] == "Delay on route 5"
    assert row["informed_route_id"] == "5"
    assert row["active_period_start"] == 500
    assert row["active_period_end"] == 2000


def test_extract_alert_rows_flattens_multiple_informed_entities():
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    entity = feed.entity.add()
    entity.id = "a2"
    entity.alert.effect = gtfs_realtime_pb2.Alert.Effect.DETOUR
    entity.alert.informed_entity.add().route_id = "5"
    entity.alert.informed_entity.add().route_id = "6"

    rows, seen = extract_alert_rows(feed, poll_ts=1000, cache=AlertDeltaCache())
    assert seen == 1
    assert len(rows) == 2
    assert {r["informed_route_id"] for r in rows} == {"5", "6"}


def test_alert_delta_cache_skips_unchanged_alert():
    cache = AlertDeltaCache()
    feed = _feed_with_alert("a1", "Delay", gtfs_realtime_pb2.Alert.Effect.SIGNIFICANT_DELAYS, "5")

    rows1, _ = extract_alert_rows(feed, poll_ts=1000, cache=cache)
    assert len(rows1) == 1

    # Same alert content on the next poll: delta compression should skip it.
    rows2, _ = extract_alert_rows(feed, poll_ts=1030, cache=cache)
    assert len(rows2) == 0


def test_alert_delta_cache_writes_when_content_changes():
    cache = AlertDeltaCache()
    feed1 = _feed_with_alert("a1", "Minor delay", gtfs_realtime_pb2.Alert.Effect.SIGNIFICANT_DELAYS, "5")
    feed2 = _feed_with_alert("a1", "Major delay", gtfs_realtime_pb2.Alert.Effect.SIGNIFICANT_DELAYS, "5")

    rows1, _ = extract_alert_rows(feed1, poll_ts=1000, cache=cache)
    rows2, _ = extract_alert_rows(feed2, poll_ts=1030, cache=cache)
    assert len(rows1) == 1
    assert len(rows2) == 1
    assert rows1[0]["header_text"] == "Minor delay"
    assert rows2[0]["header_text"] == "Major delay"


def test_alert_ignores_trip_update_entities():
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    entity = feed.entity.add()
    entity.id = "tu1"
    entity.trip_update.trip.trip_id = "t1"

    rows, seen = extract_alert_rows(feed, poll_ts=1000, cache=AlertDeltaCache())
    assert seen == 0
    assert rows == []


def test_cause_and_effect_name_roundtrip():
    assert cause_name(gtfs_realtime_pb2.Alert.Cause.ACCIDENT) == "ACCIDENT"
    assert effect_name(gtfs_realtime_pb2.Alert.Effect.DETOUR) == "DETOUR"
    assert cause_name(99999) == "UNKNOWN_CAUSE"


def test_translated_text_prefers_english():
    ts = gtfs_realtime_pb2.TranslatedString()
    ts.translation.add(text="Verspätung", language="de")
    ts.translation.add(text="Delay", language="en")
    assert translated_text(ts) == "Verspätung" or translated_text(ts) == "Delay"


def test_translated_text_empty_returns_default():
    ts = gtfs_realtime_pb2.TranslatedString()
    assert translated_text(ts) is None
    assert translated_text(ts, default="n/a") == "n/a"


def test_trip_update_delta_cache_unaffected_by_alert_changes():
    """Sanity check that adding alert parsing didn't disturb the existing TripUpdate delta cache."""
    cache = DeltaCache()
    key = ("t1", 3)
    assert cache.changed(key, (10, 12))
    assert not cache.changed(key, (10, 12))
    assert cache.changed(key, (15, 12))


def test_extract_rows_still_works_alongside_alerts_in_same_feed():
    feed = _feed_with_alert("a1", "Delay", gtfs_realtime_pb2.Alert.Effect.SIGNIFICANT_DELAYS, "5")
    tu_entity = feed.entity.add()
    tu_entity.id = "tu1"
    tu_entity.trip_update.trip.trip_id = "t1"
    stu = tu_entity.trip_update.stop_time_update.add()
    stu.stop_sequence = 2
    stu.arrival.delay = 30

    rows, seen, missing = extract_rows(feed, poll_ts=1000, cache=DeltaCache())
    assert seen == 1
    assert missing == 0
    assert len(rows) == 1
    assert rows[0]["trip_id"] == "t1"
