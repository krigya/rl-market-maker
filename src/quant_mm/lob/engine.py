"""The matching engine: the public API over the order book.

This is where **price-time priority** is enforced on every incoming order:

* *Price priority*  -- we always match against the best (most aggressive)
  opposing price level first (:meth:`OrderBook.best_level`).
* *Time priority*   -- within a level we always hit the **head** of the FIFO,
  i.e. the order that has rested longest.

An incoming order "walks the book", taking liquidity level by level until it is
either filled, or (for a limit order) its price is no longer marketable, at
which point any remainder **rests** in the book. Each crossing produces a
:class:`Trade` printed at the resting order's price.
"""

from __future__ import annotations

from collections.abc import Callable

from .book import OrderBook
from .order import Order, OrderType, Side, Trade


class MatchingEngine:
    """A single-instrument, price-time-priority matching engine."""

    def __init__(self) -> None:
        self.book = OrderBook()
        self.tape: list[Trade] = []          # chronological record of all executions
        self._seq = 0                        # logical clock -> order time priority
        self._next_id = 0                    # auto-assigned order ids
        self._on_trade: list[Callable[[Trade], None]] = []

    # ----- observation hooks ---------------------------------------------

    def subscribe(self, callback: Callable[[Trade], None]) -> None:
        """Register a callback invoked on every trade (used by the simulator/agent)."""
        self._on_trade.append(callback)

    # ----- order entry ---------------------------------------------------

    def submit(
        self,
        side: Side,
        quantity: int,
        price: float | None = None,
        order_type: OrderType = OrderType.LIMIT,
        order_id: int | None = None,
    ) -> tuple[Order, list[Trade]]:
        """Submit an order. Returns the order (with updated ``remaining``) and its fills.

        A ``LIMIT`` order requires a ``price``. A ``MARKET`` order ignores price,
        takes whatever liquidity exists, and never rests (any unfilled remainder
        is simply dropped — modelled as immediate-or-cancel for the residual).
        """
        if order_type is OrderType.LIMIT and price is None:
            raise ValueError("limit orders require a price")
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        self._seq += 1
        if order_id is None:
            order_id = self._next_id
            self._next_id += 1
        order = Order(
            id=order_id,
            side=side,
            price=price if order_type is OrderType.LIMIT else None,
            quantity=quantity,
            seq=self._seq,
        )

        trades = self._match(order, order_type)

        if order_type is OrderType.LIMIT and not order.is_filled:
            self.book.add(order)  # rest the unfilled remainder

        return order, trades

    def cancel(self, order_id: int) -> Order | None:
        """Cancel a resting order by id. Returns it, or ``None`` if not on the book."""
        return self.book.cancel(order_id)

    # ----- the matching loop ---------------------------------------------

    def _match(self, incoming: Order, order_type: OrderType) -> list[Order]:
        trades: list[Trade] = []
        opposite = incoming.side.opposite

        while not incoming.is_filled:
            best = self.book.best_level(opposite)
            if best is None:
                break  # no liquidity left to take

            # Price priority + marketability check (limit orders only).
            if order_type is OrderType.LIMIT:
                if incoming.side is Side.BID and best.price > incoming.price:
                    break  # cheapest ask is above our bid -> stop, rest the rest
                if incoming.side is Side.ASK and best.price < incoming.price:
                    break

            resting = best.head  # time priority: oldest order at this price
            assert resting is not None
            fill_qty = min(incoming.remaining, resting.remaining)

            trade = Trade(
                price=resting.price,
                quantity=fill_qty,
                seq=incoming.seq,
                aggressor_id=incoming.id,
                maker_id=resting.id,
                aggressor_side=incoming.side,
            )
            trades.append(trade)
            self.tape.append(trade)

            incoming.remaining -= fill_qty
            self.book.execute(resting, fill_qty)  # decrements/removes the maker

            for cb in self._on_trade:
                cb(trade)

        return trades

    # ----- convenience read-throughs -------------------------------------

    def best_bid(self) -> float | None:
        return self.book.best_bid()

    def best_ask(self) -> float | None:
        return self.book.best_ask()

    def mid(self) -> float | None:
        return self.book.mid()

    def spread(self) -> float | None:
        return self.book.spread()
