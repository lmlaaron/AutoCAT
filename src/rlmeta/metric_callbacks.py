from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep


class MetricCallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        info = timestep.info
        if info["is_guess"]:
            self._custom_metrics["correct_rate"] = float(info["guess_correct"])
        

class MACallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()
    
    def on_episode_start(self, index: int) -> None:
        self.tot_guess = 0
        self.acc_guess = 0
        self.tot_detect = 0
        self.acc_detect = 0

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        attacker_info = timestep['attacker'].info
        if attacker_info["is_guess"] and attacker_info['action_mask']['attacker']:
            self.tot_guess += 1
            self.acc_guess += int(attacker_info["guess_correct"])
        detector_info = timestep['detector'].info
        self.tot_detect += 1
        self.acc_detect += int(detector_info["guess_correct"])
        if timestep['detector'].done:
            self._custom_metrics["detector_correct_rate"] = self.acc_detect / self.tot_detect
            if attacker_info['action_mask']['attacker']:
                if self.tot_guess>0:
                    self._custom_metrics["attacker_correct_rate"] = self.acc_guess / float(self.tot_guess)
                self._custom_metrics["num_total_guess"] = float(self.tot_guess)

class CCHunterMetricCallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_start(self, index: int) -> None:
        self.tot_guess = 0
        self.acc_guess = 0

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        info = timestep.info

        if info["is_guess"]:
            self.tot_guess += 1
            self.acc_guess += int(info["guess_correct"])

        if timestep.done:
            self._custom_metrics["total_guess"] = self.tot_guess
            if self.tot_guess > 0:
                self._custom_metrics[
                    "correct_rate"] = self.acc_guess / self.tot_guess

        if "cc_hunter_attack" in info:
            self._custom_metrics["cc_hunter_attack"] = float(
                info["cc_hunter_attack"])

class CycloneMetricCallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_start(self, index: int) -> None:
        self.tot_guess = 0
        self.acc_guess = 0

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        info = timestep.info

        if info["is_guess"]:
            self.tot_guess += 1
            self.acc_guess += int(info["guess_correct"])

        if timestep.done:
            self._custom_metrics["total_guess"] = self.tot_guess
            if self.tot_guess > 0:
                self._custom_metrics[
                    "correct_rate"] = self.acc_guess / self.tot_guess

        if "cyclone_attack" in info:
            self._custom_metrics["cyclone_attack"] = float(
                info["cyclone_attack"])

