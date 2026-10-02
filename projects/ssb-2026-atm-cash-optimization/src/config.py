from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    max_route_minutes: float = 300.0
    service_minutes_per_stop: float = 20.0
    avg_speed_kmh: float = 35.0
    revisit_gap_days: float = 2.0          
    unrecyclable_threshold: float = 40_000.0

    min_order_amount: float = 80_000.0      
    order_rounding_step: float = 10_000.0   
    demand_window_days: int = 2             
    category_weights: dict = field(default_factory=lambda: {
        "VIP": 7, "A+": 6, "A": 5, "B": 4, "C": 3,
    })
    default_weight: int = 3              

    out_of_scope_codes: frozenset = frozenset({"K001", "K005", "K006"})

    stockout_code: str = "K010"     
    idle_removal_code: str = "K012"  

    vaka_priority_mode: str = "lowest_priority_wins"
    rescue_technical_with_alternate_case: bool = True
    k004_special_case_zero_order: bool = False
    apply_min_floor_to_idle_reload: bool = False
    strict_infeasible_reporting: bool = True
