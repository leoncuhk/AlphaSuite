"""
Contains the BaseTrader class, which encapsulates common trade execution logic
for machine learning-based strategies.
"""
import logging
from decimal import Decimal
from typing import Optional
from pybroker import ExecContext

logger = logging.getLogger(__name__)

class BaseTrader:
    """
    A base class for ML traders that buy on a specific setup condition.
    It handles the common logic of checking predictions, managing risk, and placing orders.
    """
    def __init__(self, model_name: Optional[str], params_map: dict):
        self.model_name = model_name
        self.params_map = params_map

    def _check_setup_condition(self, ctx: ExecContext, params: dict) -> bool:
        """
        Checks for the setup condition using the pre-calculated 'setup_mask' column.
        """
        # Accessing `ctx.setup_mask` triggers the `__getattr__` method in `ExecContext`.
        # This method fetches a rolling window of the 'setup_mask' data series from
        # the underlying data scope, ending at the current bar.
        # Therefore, `[-1]` correctly and safely accesses the value for the current bar,
        # not the last value of the entire historical dataset.
        setup_mask = getattr(ctx, 'setup_mask', None)
        return setup_mask[-1] if setup_mask is not None and len(setup_mask) > 0 else False

    def _get_exit_logic(self, ctx: ExecContext, params: dict) -> dict:
        """
        Returns a dictionary with exit parameters for pybroker.
        Default is an initial stop and a trailing stop based on ATR.
        """
        if not hasattr(ctx, 'atr'):
            raise ValueError("ATR indicator is required for exit logic but not found in context. "
             "Ensure 'atr' is calculated and available in the data.")

        # Check if the ATR feature has values before accessing it.
        atr = Decimal(str(ctx.atr[-1])) if ctx.atr is not None and len(ctx.atr) > 0 else Decimal('0')
        # Use str() to create Decimals from floats to avoid precision issues.
        stop_multiplier = Decimal(str(params.get('initial_stop_atr_multiplier', 2.0)))
        trail_multiplier = Decimal(str(params.get('trailing_stop_atr_multiplier', 1.5)))

        return {
            'stop_loss': atr * stop_multiplier,
            'stop_trailing': atr * trail_multiplier
        }

    def _execute_buy(self, ctx: ExecContext, params: dict):
        """
        A helper method that contains the common logic for executing a buy order,
        including risk-based position sizing and setting exit parameters.
        """
        risk_per_trade_pct = Decimal(str(params.get('risk_per_trade_pct', 0.02)))
        exit_params = self._get_exit_logic(ctx, params)
        stop_loss_points = exit_params.get('stop_loss')

        if stop_loss_points and stop_loss_points > 0:
            ctx.buy_shares = int((ctx.total_equity * risk_per_trade_pct) / stop_loss_points)
            for key, value in exit_params.items():
                if value is not None and value > 0:
                    setattr(ctx, key, value)

    def execute(self, ctx: ExecContext):
        """The core execution logic that runs on every bar."""
        params = self.params_map.get(ctx.symbol)
        if params is None:
            # This should not happen if the trader is instantiated correctly.
            # Log a warning and use an empty dict to prevent a crash.
            logger.warning(f"No parameters found for symbol '{ctx.symbol}' in trader's params_map. Using empty params.")
            params = {}

        preds = ctx.preds(self.model_name)
        if preds is None or len(preds) == 0: return

        prob_win = preds[-1][1]
        probability_threshold = params.get('probability_threshold', 0.60)
        is_setup = self._check_setup_condition(ctx, params)
        # if is_setup:
        #     print(f"{is_setup=}, {prob_win=}")
        if is_setup and prob_win > probability_threshold and not ctx.long_pos():
            self._execute_buy(ctx, params)

class RuleBasedTrader(BaseTrader):
    """
    A trader for rule-based strategies that do not use ML models.
    It enters trades based solely on the `setup_mask` condition.
    """
    def execute(self, ctx: ExecContext):
        """The core execution logic for rule-based strategies."""
        params = self.params_map.get(ctx.symbol)
        if params is None:
            logger.warning(f"No parameters found for symbol '{ctx.symbol}' in trader's params_map. Using empty params.")
            params = {}

        # For rule-based strategies, we only check the setup condition.
        is_setup = self._check_setup_condition(ctx, params)

        if is_setup and not ctx.long_pos():
            self._execute_buy(ctx, params)