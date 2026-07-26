"""Order-book invariant tests.

These assert the *defining properties* of a price-time-priority book, in
isolation from the matching loop:

* best bid/ask track the most aggressive resting prices,
* cancellation removes liquidity (and is robust to unknown ids),
* level volume bookkeeping stays consistent,
* the imbalance feature behaves sensibly.
"""

from quant_mm.lob import Order, OrderBook, Side


def rest(book: OrderBook, oid: int, side: Side, price: float, qty: int, seq: int) -> Order:
    """Helper: build a resting order and add it to the book."""
    order = Order(id=oid, side=side, price=price, quantity=qty, seq=seq)
    book.add(order)
    return order


def test_best_prices_track_most_aggressive() -> None:
    book = OrderBook()
    rest(book, 1, Side.BID, 99.0, 10, 1)
    rest(book, 2, Side.BID, 100.0, 5, 2)   # better bid
    rest(book, 3, Side.ASK, 101.0, 8, 3)
    rest(book, 4, Side.ASK, 102.0, 8, 4)

    assert book.best_bid() == 100.0
    assert book.best_ask() == 101.0
    assert book.mid() == 100.5
    assert book.spread() == 1.0


def test_empty_book_reports_none() -> None:
    book = OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid() is None
    assert book.spread() is None


def test_cancel_removes_liquidity_and_exposes_next_level() -> None:
    book = OrderBook()
    rest(book, 1, Side.BID, 100.0, 5, 1)
    rest(book, 2, Side.BID, 99.0, 7, 2)

    assert book.best_bid() == 100.0
    cancelled = book.cancel(1)
    assert cancelled is not None and cancelled.id == 1
    # With the top level gone, the next-best price surfaces.
    assert book.best_bid() == 99.0


def test_cancel_unknown_id_is_safe() -> None:
    book = OrderBook()
    rest(book, 1, Side.BID, 100.0, 5, 1)
    assert book.cancel(999) is None
    assert book.best_bid() == 100.0


def test_level_volume_aggregates_orders() -> None:
    book = OrderBook()
    rest(book, 1, Side.BID, 100.0, 5, 1)
    rest(book, 2, Side.BID, 100.0, 3, 2)  # same price -> same level
    level = book.best_level(Side.BID)
    assert level is not None
    assert level.price == 100.0
    assert level.volume == 8
    assert level.count == 2

    book.cancel(1)
    assert level.volume == 3
    assert level.count == 1


def test_imbalance_sign() -> None:
    book = OrderBook()
    rest(book, 1, Side.BID, 100.0, 30, 1)
    rest(book, 2, Side.ASK, 101.0, 10, 2)
    # More resting demand than supply -> positive imbalance.
    assert book.imbalance() > 0
    assert abs(book.imbalance() - (30 - 10) / (30 + 10)) < 1e-12
