import dataclasses
from typing import Dict


@dataclasses.dataclass
class Company:
    id: str
    name: str
    technologies: list
    tot_previous_investments: int = 0
    num_previous_investments: int = 0


@dataclasses.dataclass
class Investor:
    investor_id : str
    name: str
    announced_on: str
    raised_amount_usd: int = 0
    num_investors: int = 0



@dataclasses.dataclass
class Technology:
    name: str
    # score: float = 0
    # rank_algo: float =  0 # rank obtained using the TechRank algorithm
    # rank_analytic: float = 0 # rank obtained using w_star_analytic (needed in the parameters optimization step)
    # tot_previous_investments: int = 0

    # def update_score(self, a: int):
    #     self.score = a
