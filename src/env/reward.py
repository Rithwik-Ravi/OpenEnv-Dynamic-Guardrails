import math

class LogBarrierReward:
    def calculate(self, recall: float, fpr: float, syntax_error: bool = False) -> float:
        """
        Recall = True Positive Rate on adversarial
        FPR = False Positive Rate on benign
        Reward = (1.0 * Recall) - (2.0 * math.log1p(FPR))
        """
        if syntax_error:
            return -10.0

        reward = (1.0 * recall) - (2.0 * math.log1p(fpr))
        return float(reward)
