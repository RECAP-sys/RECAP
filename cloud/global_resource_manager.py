class GlobalResourceManager:
    """
    Cloud global resource manager.
    Maintains global resource state and calculates shadow prices.
    """

    def __init__(self, capacity_comp: float = 100.0, capacity_bw: float = 100.0):
        self.Q_comp_global: float = 0.0
        self.Q_bw_global: float = 0.0
        self.capacity_comp: float = capacity_comp
        self.capacity_bw: float = capacity_bw

    def update_queues(self, used_comp: float, used_bw: float) -> None:
        """
        Update queue lengths based on usage and capacity.
        Q(t+1) = max(0, Q(t) + used - capacity)
        """
        self.Q_comp_global = max(0.0, self.Q_comp_global + used_comp - self.capacity_comp)
        self.Q_bw_global = max(0.0, self.Q_bw_global + used_bw - self.capacity_bw)

    def get_shadow_prices(self) -> dict[str, float]:
        """
        Return the current queue lengths as shadow prices (for broadcasting).
        """
        return {"price_comp": self.Q_comp_global, "price_bw": self.Q_bw_global}
