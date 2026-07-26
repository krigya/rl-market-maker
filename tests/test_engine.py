"""Matching-engine tests: price-time priority, partial fills, market orders.

If these pass, the engine is behaving like a real exchange. They are written to
read as executable specifications of the matching rules.
"""

import pytest

from quant_mm.lob import MatchingEngine, OrderType, Side


def test_price_priority_best_price_fills_first() -> None:
    """A marketable buy should hit the cheapest ask first."""
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=5, price=101.0)  # id 0
    eng.submit(Side.ASK, quantity=5, price=100.0)  # id 1, cheaper -> should fill first

    _, trades = eng.submit(Side.BID, quantity=5, price=101.0)  # marketable buy
    assert len(trades) == 1
    assert trades[0].price == 100.0      # filled at the *cheaper* (better) ask
    assert trades[0].maker_id == 1
    assert eng.best_ask() == 101.0       # the 101 ask still rests


def test_time_priority_fifo_within_a_level() -> None:
    """At equal price, the order that arrived first fills first."""
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=5, price=100.0)  # id 0, earlier
    eng.submit(Side.ASK, quantity=5, price=100.0)  # id 1, later

    _, trades = eng.submit(Side.BID, quantity=5, price=100.0)
    assert len(trades) == 1
    assert trades[0].maker_id == 0       # the earlier order was hit
    # The later order remains untouched at the front of the queue.
    level = eng.book.best_level(Side.ASK)
    assert level is not None and level.head.id == 1


def test_partial_fill_rests_remainder() -> None:
    """A large limit order takes available liquidity then rests the rest."""
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=3, price=100.0)

    order, trades = eng.submit(Side.BID, quantity=10, price=100.0)
    assert sum(t.quantity for t in trades) == 3
    assert order.remaining == 7          # unfilled remainder...
    assert eng.best_bid() == 100.0       # ...now rests as the best bid
    book_level = eng.book.best_level(Side.BID)
    assert book_level is not None and book_level.volume == 7


def test_aggressor_partially_fills_resting_order() -> None:
    """A small aggressor leaves the resting order partly filled and still resting."""
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=10, price=100.0)  # id 0

    _, trades = eng.submit(Side.BID, quantity=4, price=100.0)
    assert trades[0].quantity == 4
    level = eng.book.best_level(Side.ASK)
    assert level is not None
    assert level.head.id == 0 and level.head.remaining == 6


def test_walks_multiple_levels() -> None:
    """A big marketable order sweeps several price levels in price order."""
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=2, price=100.0)
    eng.submit(Side.ASK, quantity=2, price=101.0)
    eng.submit(Side.ASK, quantity=2, price=102.0)

    _, trades = eng.submit(Side.BID, quantity=5, price=102.0)
    assert [t.price for t in trades] == [100.0, 101.0, 102.0]
    assert [t.quantity for t in trades] == [2, 2, 1]   # last level partially taken
    assert eng.best_ask() == 102.0                     # 1 lot left at the top level


def test_market_order_ignores_price_and_does_not_rest() -> None:
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=3, price=100.0)
    eng.submit(Side.ASK, quantity=3, price=105.0)

    order, trades = eng.submit(Side.BID, quantity=4, price=None, order_type=OrderType.MARKET)
    assert [t.price for t in trades] == [100.0, 105.0]
    assert order.remaining == 0
    # Nothing rests on the bid side from a market order.
    assert eng.best_bid() is None


def test_market_order_remainder_is_dropped_on_empty_book() -> None:
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=2, price=100.0)

    order, trades = eng.submit(Side.BID, quantity=5, price=None, order_type=OrderType.MARKET)
    assert sum(t.quantity for t in trades) == 2
    assert order.remaining == 3          # 3 lots went unfilled and were NOT rested
    assert eng.best_bid() is None


def test_non_marketable_limit_just_rests() -> None:
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=5, price=101.0)

    order, trades = eng.submit(Side.BID, quantity=5, price=100.0)  # below best ask
    assert trades == []
    assert order.remaining == 5
    assert eng.best_bid() == 100.0


def test_cancelled_order_is_skipped_by_matching() -> None:
    eng = MatchingEngine()
    a, _ = eng.submit(Side.ASK, quantity=5, price=100.0)  # id 0
    eng.submit(Side.ASK, quantity=5, price=101.0)         # id 1
    eng.cancel(a.id)

    _, trades = eng.submit(Side.BID, quantity=5, price=101.0)
    assert trades[0].price == 101.0      # the 100 ask was cancelled, so we hit 101
    assert trades[0].maker_id == 1


def test_invalid_orders_raise() -> None:
    eng = MatchingEngine()
    with pytest.raises(ValueError):
        eng.submit(Side.BID, quantity=5, price=None, order_type=OrderType.LIMIT)
    with pytest.raises(ValueError):
        eng.submit(Side.BID, quantity=0, price=100.0)


def test_tape_records_all_trades() -> None:
    eng = MatchingEngine()
    eng.submit(Side.ASK, quantity=2, price=100.0)
    eng.submit(Side.ASK, quantity=2, price=101.0)
    eng.submit(Side.BID, quantity=4, price=101.0)
    assert len(eng.tape) == 2
    assert sum(t.quantity for t in eng.tape) == 4
