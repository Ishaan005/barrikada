from dataclasses import dataclass

@dataclass
class PatternStats:
    pattern:str
    safe_df:int
    mal_df:int

    @property
    def total_df(self):
        return self.safe_df + self.mal_df
    
    @property
    def mal_precision(self):
        if self.total_df:
            return self.mal_df / self.total_df
        else:
            return 0.0
        
    @property
    def safe_precision(self):
        if self.total_df:
            return self.safe_df / (self.safe_df + self.mal_df)
        else:
            return 0.0