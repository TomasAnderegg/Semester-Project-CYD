import dataclasses
from typing import Dict


@dataclasses.dataclass
class Company:
    id: str
    name: str
    location: Dict[str, str] # location divided in parts (city, state...)
    technologies: list
    lat: float # latitude of the location
    lon: float # longitude of the location
    degree: float =  0
    rank_CB: float = 0
    rank_algo: float =  0 # rank obtained using the TechRank algorithm
    rank_analytic: float = 0 # rank obtained using w_star_analytic (needed in the parameters optimization step)
    tot_previous_investments: int = 0
    num_previous_investments: int = 0


@dataclasses.dataclass
class Investor:
    id: str
    name: str
    location: Dict[str, str]
    investor_type: str
    investment_count: int = 0 # number of investments
    lat: float =  0 # latitude of the location
    lon: float =  0 # longitude of the location


@dataclasses.dataclass
class Technology:
    name: str
    # score: float = 0
    degree: float =  0
    rank_algo: float =  0 # rank obtained using the TechRank algorithm
    rank_analytic: float = 0 # rank obtained using w_star_analytic (needed in the parameters optimization step)
    tot_previous_investments: int = 0

    def update_score(self, a: int):
        self.score = a
