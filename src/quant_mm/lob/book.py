"""The order book: price levels plus best-price bookkeeping.

Data-structure choices (the part worth defending in an interview)
----------------------------------------------------------------
A limit order book must support, fast:

1. **Best bid / best ask** lookup            -> read the top of book every tick.
2. **Insert** a resting order at a price       -> add liquidity.
3. **Cancel** an arbitrary resting order       -> pull liquidity.
4. **Match** the front order at the best price  -> price-time priority.

We get all four cheaply with two structures working together:

* ``PriceLevel`` - an intrusive doubly-linked **FIFO** of orders at one price.
  Append (new order) and unlink (cancel/fill) are both O(1). FIFO ordering *is*
  time priority: the order that arrived first sits at the head and fills first.

* A binary **heap** of active prices per side (max-heap for bids via price
  negation, min-heap for asks). The best price is the heap root -> O(1) to read,
  O(log n) to insert a brand-new price level. We use **lazy deletion**: when a
  level empties we just drop it from the dict; its stale price lingers in the
  heap until it surfaces at the root, where we discard it. This keeps cancels
  O(1) (no heap surgery) at the cost of the occasional pop on read.

A dict ``order_id -> Order`` makes cancellation a direct O(1) lookup.
"""

from __future__ import annotations

import heapq

from .order import Order, Side


class PriceLevel:
    """A FIFO queue of resting orders sharing one price (intrusive linked list)."""

    __slots__ = ("price", "head", "tail", "volume", "count")

    def __init__(self, price: float) -> None:
        self.price = price
        self.head: Order | None = None   # earliest order (fills first)
        self.tail: Order | None = None   # latest order
        self.volume = 0                  # sum of `remaining` across the level
        self.count = 0                   # number of resting orders

    @property
    def is_empty(self) -> bool:
        return self.head is None

    def append(self, order: Order) -> None:
        """Add an order to the back of the queue (latest in time priority)."""
        order.prev = self.tail
        order.next = None
        if self.tail is None:
            self.head = order
        else:
            self.tail.next = order
        self.tail = order
        self.volume += order.remaining
        self.count += 1

    def remove(self, order: Order) -> None:
        """Unlink a known order in O(1) (used for both cancels and full fills)."""
        if order.prev is not None:
            order.prev.next = order.next
        else:  # order was the head
            self.head = order.next
        if order.next is not None:
            order.next.prev = order.prev
        else:  # order was the tail
            self.tail = order.prev
        order.prev = order.next = None
        self.volume -= order.remaining
        self.count -= 1


class OrderBook:
    """Two sides of resting liquidity with fast best-price access."""

    def __init__(self) -> None:
        self._levels: dict[Side, dict[float, PriceLevel]] = {Side.BID: {}, Side.ASK: {}}
        # Lazy-deletion price heaps. Bids negated so the max price is the root.
        self._bid_heap: list[float] = []
        self._ask_heap: list[float] = []
        self.order_map: dict[int, Order] = {}

    # ----- inserting / removing resting liquidity -------------------------

    def add(self, order: Order) -> None:
        """Rest a (partially filled or fresh) limit order in the book."""
        assert order.price is not None, "market orders never rest"
        levels = self._levels[order.side]
        level = levels.get(order.price)
        if level is None:
            level = PriceLevel(order.price)
            levels[order.price] = level
            if order.side is Side.BID:
                heapq.heappush(self._bid_heap, -order.price)
            else:
                heapq.heappush(self._ask_heap, order.price)
        level.append(order)
        self.order_map[order.id] = order

    def cancel(self, order_id: int) -> Order | None:
        """Remove a resting order by id. Returns the order, or ``None`` if absent."""
        order = self.order_map.pop(order_id, None)
        if order is None:
            return None
        level = self._levels[order.side][order.price]
        level.remove(order)
        if level.is_empty:
            del self._levels[order.side][order.price]  # stale heap entry cleaned lazily
        return order

    def execute(self, order: Order, qty: int) -> None:
        """Apply a fill of ``qty`` to a resting order, removing it if fully filled."""
        level = self._levels[order.side][order.price]
        order.remaining -= qty
        level.volume -= qty
        if order.is_filled:
            self.order_map.pop(order.id, None)
            level.remove(order)  # remaining is now 0, so this is a no-op on volume
            if level.is_empty:
                del self._levels[order.side][order.price]

    # ----- top-of-book queries -------------------------------------------

    def best_level(self, side: Side) -> PriceLevel | None:
        """Return the most aggressive non-empty price level on ``side`` (O(1) amortized)."""
        levels = self._levels[side]
        heap = self._bid_heap if side is Side.BID else self._ask_heap
        while heap:
            price = -heap[0] if side is Side.BID else heap[0]
            level = levels.get(price)
            if level is not None and not level.is_empty:
                return level
            heapq.heappop(heap)  # discard stale price
        return None

    def best_bid(self) -> float | None:
        level = self.best_level(Side.BID)
        return level.price if level else None

    def best_ask(self) -> float | None:
        level = self.best_level(Side.ASK)
        return level.price if level else None

    def mid(self) -> float | None:
        """Mid-price = midpoint of best bid and best ask (``None`` if one side empty)."""
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def spread(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, side: Side, n_levels: int = 5) -> list[tuple[float, int]]:
        """Top ``n_levels`` of ``side`` as ``(price, volume)``, best price first.

        Used to build the *book-imbalance* feature for the RL agent. Sorting the
        active prices on demand is O(L log L); the book is shallow in simulation
        so this is cheap, and it keeps the hot path (best_level) heap-fast.
        """
        levels = self._levels[side]
        prices = sorted(levels, reverse=(side is Side.BID))
        return [(p, levels[p].volume) for p in prices[:n_levels]]

    def imbalance(self, n_levels: int = 5) -> float:
        """Order-book imbalance in ``[-1, 1]``: (bid_vol - ask_vol)/(bid_vol + ask_vol).

        Positive = more resting demand than supply (mild upward pressure). This
        is one of the most predictive cheap microstructure features.
        """
        bid_vol = sum(v for _, v in self.depth(Side.BID, n_levels))
        ask_vol = sum(v for _, v in self.depth(Side.ASK, n_levels))
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total
