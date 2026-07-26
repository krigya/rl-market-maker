"""Order and trade domain types for the matching engine.

Design notes (interview-relevant)
----------------------------------
* ``Order`` doubles as a node in an *intrusive* doubly-linked list. Each price
  level in the book is a FIFO queue of orders, and we hold that queue directly
  on the orders via ``prev``/``next`` pointers. This is what real matching
  engines do: it makes **cancellation O(1)** (unlink a known node) instead of
  the O(n) you'd pay scanning a Python ``list``/``deque`` to find the order.
* Time priority is encoded by ``seq`` (a strictly increasing sequence number
  assigned by the engine), not wall-clock time. Determinism > timestamps: the
  same script always produces the same book, which matters for reproducible RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(Enum):
    """Which side of the book an order sits on."""

    BID = "bid"  # an order to BUY
    ASK = "ask"  # an order to SELL

    @property
    def opposite(self) -> "Side":
        """The side an aggressor of this order would match against."""
        return Side.ASK if self is Side.BID else Side.BID


class OrderType(Enum):
    """Execution style of an incoming order."""

    LIMIT = "limit"    # rests in the book if not fully filled
    MARKET = "market"  # takes liquidity at any price; never rests


@dataclass(slots=True)
class Order:
    """A single order, and simultaneously a node in a price level's FIFO list.

    Parameters
    ----------
    id:
        Unique identifier, used for cancellation lookups.
    side:
        :class:`Side.BID` (buy) or :class:`Side.ASK` (sell).
    price:
        Limit price. ``None`` for a market order (matches at any price).
    quantity:
        Original size submitted (immutable record).
    seq:
        Engine-assigned sequence number; lower ``seq`` = earlier in time
        priority at a given price.
    remaining:
        Unfilled size. Starts equal to ``quantity`` and decreases as the order
        fills. ``remaining == 0`` means fully filled.
    prev, next:
        Intrusive linked-list pointers within a price level. Managed by the
        book; not set by callers.
    """

    id: int
    side: Side
    price: float | None
    quantity: int
    seq: int = 0
    remaining: int = field(default=0)
    prev: "Order | None" = field(default=None, repr=False, compare=False)
    next: "Order | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.remaining == 0:
            self.remaining = self.quantity

    @property
    def is_filled(self) -> bool:
        return self.remaining == 0


@dataclass(slots=True, frozen=True)
class Trade:
    """An execution produced when an aggressing order meets a resting order.

    The trade always prints at the **resting (maker) order's price** — that is
    the essence of price priority: the order that was in the book first sets the
    price, and the incoming aggressor pays it.
    """

    price: float
    quantity: int
    seq: int            # engine sequence number at which the trade occurred
    aggressor_id: int   # the incoming (taker) order
    maker_id: int       # the resting (maker) order that was hit
    aggressor_side: Side

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Trade(seq={self.seq}, {self.aggressor_side.value} aggressor "
            f"#{self.aggressor_id} hit #{self.maker_id}: "
            f"{self.quantity} @ {self.price})"
        )
